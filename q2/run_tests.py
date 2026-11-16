#code is just to make gifs and plots for the Q2 navigation policy



import os, sys, argparse, pickle
from multiprocessing import get_context
START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"  # Windows =spawn
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "plots", "gifs"):             
    sys.path.insert(0, os.path.join(HERE, sub))
os.chdir(HERE)         
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from navigation_env import MAX_NAV_STEPS, CLEARANCE, DEPTH_BINS, FAR
from q2_navigation_env import Q2NavigationEnv, MAPS, GOAL, TESTS   # maps + 5 test starts shared with training

PRESETS = {
    "final":    dict(nav="models/navigation_policy.zip", target_height=4.0, render_far=20.0, depth_max=20.0),
    "original": dict(nav="models/navigation_policy_before_q2.zip", target_height=0.8, render_far=0, depth_max=0),
    "tall":     dict(nav="models/navigation_policy_before_q2.zip", target_height=4.0, render_far=0, depth_max=0),
    "tall_far": dict(nav="models/navigation_policy_before_q2.zip", target_height=4.0, render_far=15.0, depth_max=0),
}


def config_from_preset(preset="final", map_name="standard", walker="models/walking_policy.zip", name=None, **override):
    """Everything a test run needs, as one picklable dict (workers get it with each job)."""
    c = dict(PRESETS[preset])
    c.update({k: v for k, v in override.items() if v is not None})
    c["render_far"], c["depth_max"] = c["render_far"] or None, c["depth_max"] or None
    name = name or (preset if map_name == "standard" else f"{map_name}_{preset}")
    title = (f"{map_name.replace('_', '-')} map" + (f", {c['target_height']:g} m goal pillar" if c["target_height"] != 0.8 else "")
             + (f", colour range {c['render_far']:g} m" if c["render_far"] else "")
             + (f", depth 0.28-{c['depth_max']:g} m" if c["depth_max"] else ""))
    return dict(c, map=map_name, walker=walker, name=name, title=title,
                plot_dir=os.path.join(HERE, "plots", name), gif_dir=os.path.join(HERE, "gifs", name))


# one test
_ctx = {}

def _setup(cfg):
    """One env + policy per worker process (seeded obstacle pool, see q2_navigation_env)."""
    if "env" not in _ctx:
        _ctx["env"] = Q2NavigationEnv(cfg["walker"], max_obstacles=20, target_height=cfg["target_height"],
                                 render_far=cfg["render_far"],
                                 depth_range=(0.28, cfg["depth_max"]) if cfg["depth_max"] else None)
        _ctx["model"] = PPO.load(cfg["nav"], device="cpu")
    return _ctx["model"], _ctx["env"]


def check_map(env, map_name):
    """The map's obstacles, after checking the goal and every start keep the training clearance."""
    layout = [(i, np.array(xy), yaw) for i, xy, yaw in MAPS[map_name]]
    for name, pt in [("goal", GOAL)] + [(f"start {k + 1}", np.array(s)) for k, (s, _) in enumerate(TESTS)]:
        gap = min(env._gap(pt, env.obstacles[i], xy, yaw) for i, xy, yaw in layout)
        assert gap > CLEARANCE, f"{name} only {gap:.2f} m from an obstacle"
    return layout


def fixed_reset(env, map_name, start, heading_deg, seed):
    """NavEnv.reset with this map, goal, start and heading instead of random ones."""
    layout = check_map(env, map_name)
    env._random_layout = lambda: (np.array(start), GOAL.copy(), layout)
    env.reset(seed=seed)
    env.walker_obs, _ = env.walk.reset(seed=seed, options={"pose": (*start, np.radians(heading_deg))})
    env.walk.command = np.zeros(3)
    env.walker_obs = env.walk._obs()
    env.prev_dist = env._target_dist()
    env.cam.resetBasePositionAndOrientation(env.target_id, [*GOAL, env.target_height / 2], [0, 0, 0, 1])
    return env._obs()


def nearest_obstacle(env, reach=2.0):
    """True gap from any robot link to the nearest obstacle / wall (for plotting; the reward only
    sees it up to SAFE_DIST)."""
    gaps = [pt[8] for b in env._solid_ids()
            for pt in env.walk.pb.getClosestPoints(env.walk.robot, b, distance=reach)]
    return min(gaps, default=reach)


def pose(env):
    """Robot body position (x, y) and heading (rad), read from PyBullet."""
    pos, quat = env.walk.pb.getBasePositionAndOrientation(env.walk.robot)
    return np.array(pos[:2]), p.getEulerFromQuaternion(quat)[2]


def outcome_of(info):
    """How the episode ended: success / collision / fell / timeout."""
    return ("success" if info["success"] else "collision" if info["collision"]
            else "fell" if info["fell"] else "timeout")


def run_episode(env, model, k, cfg, on_step=None):
    """Test k: drive the policy until goal / contact / fall / 60 s; record every 0.1 s step.
    on_step(env, rec, step, done, outcome) is called after each step (the GIF recorder uses it)."""
    start, heading = TESTS[k]
    obs = fixed_reset(env, cfg["map"], start, heading, seed=k)
    rec = dict(t=[0.0], xy=[env._robot_xy()], yaw=[pose(env)[1]], dist=[env._target_dist()],
               clear=[nearest_obstacle(env)], cmd=[np.zeros(3)], seen=[bool(env.target_visible)], reward=[0.0])
    snap = None
    for step in range(1, MAX_NAV_STEPS + 1):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        xy, yaw = pose(env)
        rec["t"].append(step * 0.1); rec["xy"].append(xy); rec["yaw"].append(yaw)
        rec["dist"].append(env._target_dist()); rec["clear"].append(nearest_obstacle(env))
        rec["cmd"].append(env.walk.command.copy()); rec["seen"].append(bool(env.target_visible))
        rec["reward"].append(rec["reward"][-1] + r)
        # one camera snapshot for the observation figure: goal in view and an obstacle close
        if snap is None and env.target_visible and min(obs[:DEPTH_BINS]) * env.depth_max < 1.75:
            rgb, depth = env._camera()
            snap = dict(rgb=rgb.astype(np.uint8), depth=depth, bins=obs[:DEPTH_BINS].copy(),
                        target=obs[DEPTH_BINS:DEPTH_BINS + 4].copy(), t=step * 0.1, dmax=env.depth_max)
        done = term or trunc
        if on_step:
            on_step(env, rec, step, done, outcome_of(info))
        if done:
            break
    rec = {key: np.array(v) for key, v in rec.items()}
    rec.update(k=k, outcome=outcome_of(info), snap=snap,
               layout=[(dict(kind=env.obstacles[i]["kind"], half=list(env.obstacles[i]["half"]),
                             round=env.obstacles[i]["round"]), np.array(xy0), yaw0) for i, xy0, yaw0 in env.layout])
    return rec


def run_test(job):
    """Pool worker: one test, optionally recording its GIF (code in gifs/make_gifs.py)."""
    k, cfg, gifs = job
    model, env = _setup(cfg)
    recorder = None
    if gifs:
        from make_gifs import GifRecorder
        recorder = GifRecorder(env, cfg["target_height"])
    rec = run_episode(env, model, k, cfg, on_step=recorder)
    if recorder:
        os.makedirs(cfg["gif_dir"], exist_ok=True)
        recorder.save(os.path.join(cfg["gif_dir"], f"q2_test{k + 1}.gif"))
    return rec


def run_all(cfg, gifs=True):
    """The 5 tests in parallel (one process each)."""
    with get_context(START_METHOD).Pool(len(TESTS)) as pool:
        return pool.map(run_test, [(k, cfg, gifs) for k in range(len(TESTS))])


def save_results(R, cfg):
    """Print the results table, save it (results/<name>_results.txt) and the recorded runs (<name>_runs.pkl)."""
    rows = ["test  start          heading  result     time(s)  path(m)  straight(m)  min clearance(m)"]
    for r in R:
        path = np.sum(np.linalg.norm(np.diff(r["xy"], axis=0), axis=1))
        rows.append(f"{r['k'] + 1:4d}  ({TESTS[r['k']][0][0]:+.1f}, {TESTS[r['k']][0][1]:+.1f})  "
                    f"{TESTS[r['k']][1]:5d}°   {r['outcome']:9s}  {r['t'][-1]:7.1f}  {path:7.2f}  "
                    f"{np.linalg.norm(GOAL - r['xy'][0]):11.2f}  {r['clear'].min():16.2f}")
    rows.append(f"reached the goal: {sum(r['outcome'] == 'success' for r in R)} / {len(R)}   "
                f"(nav {cfg['nav']}, walker {cfg['walker']})")
    print("\n".join(rows))
    rows.insert(0, f"map: {cfg['map']}, goal pillar {cfg['target_height']:g} m, colour render range "
                   f"{cfg['render_far'] or FAR:g} m, depth range "
                   f"{'0.28-%g' % cfg['depth_max'] if cfg['depth_max'] else '0.1-5'} m")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{cfg['name']}_results.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(HERE, "results", f"{cfg['name']}_runs.pkl"), "wb") as f:
        pickle.dump(dict(runs=R, title=cfg["title"], config=cfg), f)


def main():
    """Command line: pick the setup, run the 5 tests, save the results, then make the plots and GIFs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="final", choices=list(PRESETS),
                    help="final = the Q2 result (default); original / tall / tall_far = earlier setups")
    ap.add_argument("--map", default="standard", choices=list(MAPS))
    ap.add_argument("--nav", help="navigation policy zip (default: from the preset)")
    ap.add_argument("--walker", default="models/walking_policy.zip")
    ap.add_argument("--target-height", type=float, help="goal pillar height (m)")
    ap.add_argument("--render-far", type=float, help="colour render range (m); 0 = original 5 m clip")
    ap.add_argument("--depth-max", type=float, help="depth range 0.28 m .. this (m); 0 = original 0.1-5 m")
    ap.add_argument("--name", help="output name (default: the preset name, e.g. final)")
    ap.add_argument("--no-gifs", action="store_true", help="skip the GIFs")
    args = ap.parse_args()
    cfg = config_from_preset(args.preset, args.map, args.walker, name=args.name, nav=args.nav,
                             target_height=args.target_height, render_far=args.render_far, depth_max=args.depth_max)

    R = run_all(cfg, gifs=not args.no_gifs)
    save_results(R, cfg)

    from make_plots import make_all
    print("plots:", ", ".join(make_all(R, cfg["title"], cfg["plot_dir"])))
    if not args.no_gifs:
        from make_gifs import all_tests_gif
        all_tests_gif(R, os.path.join(cfg["gif_dir"], "q2_all_tests.gif"))
        print("gifs: ", ", ".join(f"{g} ({os.path.getsize(os.path.join(cfg['gif_dir'], g)) / 1e6:.1f} MB)"
                                  for g in sorted(os.listdir(cfg["gif_dir"]))))


if __name__ == "__main__":
    main()
