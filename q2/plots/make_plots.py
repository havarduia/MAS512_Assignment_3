"""All Q2 plots, drawn from the recorded test runs (no simulation needed).

usage: python plots/make_plots.py [--name final]
       reads results/<name>_runs.pkl (written by run_tests.py) and the training logs in logs/,
       writes plots/<name>/:

  q2_trajectories.png   the 5 paths on the map (the comparison asked for in Q2)
  q2_distance.png       distance to the goal over time, per test
  q2_clearance.png      distance to the nearest obstacle over time (collision avoidance)
  q2_commands.png       the navigation policy's velocity commands over time
  q2_observation.png    what the policy sees: camera, depth, the 16 depth bins, red mask
  q2_training.png       navigation training: reward, success rate, obstacle curriculum
"""
import os, sys, re, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
Q2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, Q2)
import numpy as np
from plot_style import (plt, map_axes, TEST_COLORS, SURFACE, INK, INK2, MUTED, GRID, CRITICAL)
from navigation_env import ARENA, REACH_DIST, SAFE_DIST, NAV_RANGE, CAM_H, DEPTH_BINS
from q2_navigation_env import GOAL

TRAINING_LOGS = [                       # the final policy's training history, in order
    ("v1, easy map", "nav_1_easy_map.log"), ("v1, + margin penalty", "nav_2_margin_penalty.log"),
    ("v1, no blind reversing", "nav_3_no_blind_reversing_v1.log"), ("v2, hard map", "nav_4_hard_map_v2.log"),
    ("Q2: 4 m goal, crash -30", "nav_5_q2_tall_goal_crash30.log"),
    ("depth 20 m", "nav_6_q2_depth20m.log"), ("+8M", "nav_7_q2_depth20m_continued.log")]


def plot_trajectories(R, title, out):
    """The 5 paths on the map in their test colours, with start dots and the outcome in the legend."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    map_axes(ax, R[0]["layout"], GOAL, ARENA, REACH_DIST)
    for r in R:
        c = TEST_COLORS[r["k"]]
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=2.2, zorder=5,
                label=f"test {r['k'] + 1}: {'reached' if r['outcome'] == 'success' else r['outcome']} "
                      f"in {r['t'][-1]:.1f} s")
        ax.scatter(*r["xy"][0], s=70, color=SURFACE, edgecolor=c, linewidth=2.2, zorder=6)
        ax.text(r["xy"][0, 0], r["xy"][0, 1] - 0.32, f"S{r['k'] + 1}", ha="center", va="top", fontsize=9,
                color=INK2, zorder=7)
        ax.scatter(*r["xy"][-1], s=36, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=6)
    ax.text(GOAL[0], GOAL[1] + 0.6, "goal", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID,
              bbox_to_anchor=(1.0, 0.0))
    ok = sum(r["outcome"] == "success" for r in R)
    ax.set_title(f"{ok} of {len(R)} test runs reached the goal\n{title}\n"
                 f"open dot = start, red pillar = goal, dashed circle = 0.5 m reach", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "grey: boxes / pillars / walls (light) • gold: coloured decoy pillars the policy must ignore",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out, "q2_trajectories.png"), dpi=150); plt.close(fig)


def plot_lines(R, key, ylabel, title, path, hline=None, hlabel=None, ylim=None):
    """One value over time for every test (distance to the goal, or gap to the nearest obstacle)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for r in R:
        ax.plot(r["t"], r[key], color=TEST_COLORS[r["k"]], lw=1.8, label=f"test {r['k'] + 1}")
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1.2, ls="--")
        ax.text(max(r["t"][-1] for r in R), hline, f"  {hlabel}", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=5, fontsize=8.5, loc="upper right")
    ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_commands(R, out):
    """The velocity commands the navigation policy sent (forward, sideways, turn) over time."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
    for ax, j, name in zip(axes, range(3), ["forward (m/s)", "sideways (m/s)", "turn rate (rad/s)"]):
        for r in R:
            ax.plot(r["t"], r["cmd"][:, j], color=TEST_COLORS[r["k"]], lw=1.3, label=f"test {r['k'] + 1}")
        ax.axhspan(NAV_RANGE[j, 0], NAV_RANGE[j, 1], color=GRID, alpha=0.35, lw=0)
        ax.set_ylabel(name)
    axes[0].legend(ncol=5, fontsize=8.5, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Navigation policy output = velocity command sent to the walking policy "
                      "(shaded = allowed range)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "q2_commands.png"), dpi=150); plt.close(fig)


def plot_observation(R, out):
    """What the policy sees at one moment: camera image, depth image, red mask, the 16 depth bins."""
    pick = next((r for r in R if r["snap"] is not None), None)
    if pick is None:
        return
    snap = pick["snap"]
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    ax = fig.add_subplot(gs[0]); ax.imshow(snap["rgb"]); ax.set_title("camera RGB (64 × 40)", loc="left")
    ax2 = fig.add_subplot(gs[1]); im = ax2.imshow(snap["depth"], cmap="Blues_r", vmin=0, vmax=snap["dmax"])
    ax2.axhline(CAM_H // 2, color=CRITICAL, lw=1, ls="--")
    ax2.set_title("depth (m); used: rows above ---", loc="left")
    fig.colorbar(im, ax=ax2, fraction=0.035)
    r, g, b = [snap["rgb"][..., i].astype(float) for i in range(3)]
    mask = (r > 80) & (r > 2 * g) & (r > 2 * b)
    ax3 = fig.add_subplot(gs[2]); ax3.imshow(mask, cmap="Reds"); ax3.set_title("red mask → target", loc="left")
    for a in (ax, ax2, ax3):
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    ax4 = fig.add_subplot(gs[3])
    ax4.bar(np.arange(DEPTH_BINS), snap["bins"] * snap["dmax"], color=TEST_COLORS[0], width=0.8)
    ax4.set_xlabel("column band (left → right)"); ax4.set_ylabel("nearest obstacle (m)")
    vis, bearing, size, dist = snap["target"]
    ax4.set_title(f"16 depth bins  |  target: bearing {bearing:+.2f}, size {size:.2f}, dist {dist * snap['dmax']:.1f} m",
                  loc="left", fontsize=9.5)
    ax4.set_ylim(0, snap["dmax"])
    fig.suptitle(f"What the navigation policy sees (test {pick['k'] + 1}, t = {snap['t']:.1f} s)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "q2_observation.png"), dpi=150); plt.close(fig)


def read_log(path):
    """Stable-Baselines3 console log -> list of {metric: value} per update."""
    rows, cur = [], {}
    for line in open(path):
        m = re.match(r"\|\s+([\w/]+)\s+\|\s+([-\d.e+]+)", line)
        if m:
            cur[m.group(1)] = float(m.group(2))
            if m.group(1) == "total_timesteps":
                rows.append(dict(cur))
    return rows


def plot_training(out, log_dir=os.path.join(Q2, "logs")):
    """Training curves (reward, success rate, obstacle count) over the whole navigation training."""
    missing = [f for _, f in TRAINING_LOGS if not os.path.exists(os.path.join(log_dir, f))]
    if missing:                                     # never replace a good chart with an empty / partial one
        print(f"q2_training.png not redrawn: missing logs in {log_dir}: {', '.join(missing)}")
        return
    parts = [(name, read_log(os.path.join(log_dir, f))) for name, f in TRAINING_LOGS]
    parts = [(n, r) for n, r in parts if r]
    for i in range(len(parts) - 1):                 # cut each run where the next one resumed
        nxt = parts[i + 1][1][0]["total_timesteps"]
        parts[i] = (parts[i][0], [r for r in parts[i][1] if r["total_timesteps"] < nxt])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True)
    for n, (name, rows) in enumerate(parts):
        if not rows:
            continue
        x = np.array([r["total_timesteps"] for r in rows]) / 1e6
        axes[0].plot(x, [r.get("ep_rew_mean", np.nan) for r in rows], color=TEST_COLORS[0], lw=1.4)
        axes[1].plot(x, [100 * r.get("success_rate", np.nan) for r in rows], color=TEST_COLORS[0], lw=1.4)
        axes[2].step(x, [r.get("max_obstacles", np.nan) for r in rows], color=TEST_COLORS[0], lw=1.6, where="post")
        for ax in axes:
            ax.axvline(x[0], color=GRID, lw=1)
        axes[0].text(x[0], 1.02 + 0.09 * (n % 2), f" {name}", transform=axes[0].get_xaxis_transform(),
                     fontsize=8, color=INK2)
    axes[0].set_ylabel("episode reward"); axes[1].set_ylabel("success rate (%)")
    fig.text(0.01, 0.005, "Reward drops in the Q2 part because a crash costs -30 there (was -10); "
             "success rate is the comparable measure.", fontsize=8, color=MUTED)
    axes[2].set_ylabel("max obstacles\n(curriculum)"); axes[2].set_xlabel("training steps (millions, navigation)")
    axes[0].set_title("Navigation policy training (PPO): curriculum +2 obstacles each time success > 80%, "
                      "then the Q2 fine-tune", loc="left", pad=26)
    fig.tight_layout(); fig.savefig(os.path.join(out, "q2_training.png"), dpi=150); plt.close(fig)


def make_all(R, title, out):
    """Every Q2 plot for the recorded runs R into the folder out."""
    os.makedirs(out, exist_ok=True)
    plot_trajectories(R, title, out)
    plot_lines(R, "dist", "distance to goal (m)", "Distance to the goal over time (success = below 0.5 m)",
               os.path.join(out, "q2_distance.png"), hline=REACH_DIST, hlabel="reach 0.5 m", ylim=(0, None))
    plot_lines(R, "clear", "nearest obstacle (m)", "Collision avoidance: gap between the robot and the nearest "
               "obstacle (0 = contact)", os.path.join(out, "q2_clearance.png"), hline=SAFE_DIST,
               hlabel="'near' penalty below 0.35 m", ylim=(0, 2.05))
    plot_commands(R, out)
    plot_observation(R, out)
    plot_training(out)
    return sorted(f for f in os.listdir(out) if f.endswith(".png"))


def main():
    """Command line: redraw every plot from a saved run (results/<name>_runs.pkl)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="final", help="run name: reads results/<name>_runs.pkl")
    args = ap.parse_args()
    with open(os.path.join(Q2, "results", f"{args.name}_runs.pkl"), "rb") as f:
        saved = pickle.load(f)
    files = make_all(saved["runs"], saved["title"], os.path.join(HERE, args.name))
    print("plots:", ", ".join(files))


if __name__ == "__main__":
    main()
