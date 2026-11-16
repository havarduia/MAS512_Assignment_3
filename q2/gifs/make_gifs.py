"""All Q2 GIFs.

usage: python gifs/make_gifs.py [--preset final]
       replays the 5 tests (deterministic, so identical to run_tests.py) with rendering on and
       writes gifs/<name>/:

  q2_test<k>.gif     per test: 2D top-down map (obstacles, goal + 0.5 m reach circle, path so far,
                     robot heading, camera field of view) | 3D chase view with the robot camera inset
  q2_all_tests.gif   all 5 tests on one top-down map, played at the same time

run_tests.py uses the same code (GifRecorder, all_tests_gif) while it runs the tests, so a
normal test run also makes the GIFs in one pass. Frames every 0.2 s of sim time, played at
10 fps (2x speed); the last frame is held for 1.5 s.
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
Q2 = os.path.dirname(HERE)
sys.path.insert(0, Q2)
sys.path.insert(0, os.path.join(Q2, "plots"))
import numpy as np
from PIL import Image
import pybullet as p
from matplotlib.patches import Wedge, FancyArrow
from plot_style import plt, map_axes, TEST_COLORS, SURFACE, INK, SERIES, OUTCOME_COLOR
from navigation_env import ARENA, REACH_DIST, CAM_W, CAM_H, CAM_VFOV, TARGET_RADIUS, TARGET_RGB

HFOV = np.degrees(2 * np.arctan(np.tan(np.radians(CAM_VFOV) / 2) * CAM_W / CAM_H))   # camera, ~63 deg
CHASE_W, CHASE_H = 640, 480


# ================= 3D view =================
def chase_frame(env, target_vis, target_height):
    """Third-person view from behind the robot + the robot's own camera image in the top-left corner.
    target_vis: a visual-only copy of the goal in the physics client (the real goal lives in the
    camera's render client)."""
    pb = env.walk.pb
    pb.resetBasePositionAndOrientation(target_vis, [*env.target_xy, target_height / 2], [0, 0, 0, 1])
    pos, quat = pb.getBasePositionAndOrientation(env.walk.robot)
    yaw = p.getEulerFromQuaternion(quat)[2]
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    eye = np.array(pos) - 2.2 * heading + [0, 0, 1.3]                 # behind and above the robot
    view = pb.computeViewMatrix(eye, np.array(pos) + 0.8 * heading, [0, 0, 1])
    proj = pb.computeProjectionMatrixFOV(60, CHASE_W / CHASE_H, 0.1, 30)
    img = pb.getCameraImage(CHASE_W, CHASE_H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    frame = np.reshape(img, (CHASE_H, CHASE_W, 4))[:, :, :3].astype(np.uint8)
    rgb, _ = env._camera()
    inset = np.repeat(np.repeat(rgb.astype(np.uint8), 3, axis=0), 3, axis=1)   # 3x upscale
    h, w = inset.shape[:2]
    frame[2:h + 6, 2:w + 6] = 0                                          # black border
    frame[4:4 + h, 4:4 + w] = inset
    return frame


# ================= 2D view =================
class TopDown:
    """Top-down map, drawn once per episode; only the moving parts are redrawn each frame."""
    def __init__(self, env, size_px=480):
        """Draw the static map once (arena, obstacles, goal) and prepare the moving parts."""
        self.fig, self.ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
        self.fig.subplots_adjust(0.02, 0.02, 0.98, 0.9)
        layout = [(env.obstacles[i], xy, yaw) for i, xy, yaw in env.layout]
        map_axes(self.ax, layout, env.target_xy, ARENA, REACH_DIST)
        self.path, = self.ax.plot([], [], color=SERIES[0], lw=2, zorder=5)
        self.start = self.fov = self.robot = None
        self.title = self.ax.set_title("", fontsize=11, color=INK, loc="left")

    def draw(self, xy_hist, yaw, t, dist, status=None):
        """Update path, robot arrow, camera view cone and title; return the map as an image."""
        ax, xy = self.ax, np.array(xy_hist)
        self.path.set_data(xy[:, 0], xy[:, 1])
        if self.start is None:
            self.start = ax.scatter(*xy[0], s=50, color=SURFACE, edgecolor=SERIES[0], linewidth=2, zorder=6)
        for art in (self.fov, self.robot):
            if art is not None:
                art.remove()
        # camera field of view (what the policy can see), drawn 1.5 m long
        self.fov = ax.add_patch(Wedge(xy[-1], 1.5, np.degrees(yaw) - HFOV / 2, np.degrees(yaw) + HFOV / 2,
                                      color=SERIES[0], alpha=0.12, lw=0, zorder=3))
        self.robot = ax.add_patch(FancyArrow(*(xy[-1] - 0.2 * np.r_[np.cos(yaw), np.sin(yaw)]),
                                             0.4 * np.cos(yaw), 0.4 * np.sin(yaw), width=0.12,
                                             head_width=0.3, head_length=0.18, length_includes_head=True,
                                             color=INK, zorder=7))
        if status:
            self.title.set_text(f"{status.upper()}   t = {t:4.1f} s")
            self.title.set_color(OUTCOME_COLOR.get(status, INK))
        else:
            self.title.set_text(f"top-down map   t = {t:4.1f} s   target {dist:.1f} m away")
            self.title.set_color(INK)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        """Free the matplotlib figure."""
        plt.close(self.fig)


def side_by_side(left, right, scale=0.75):
    """[top-down map | 3D view] in one frame, scaled down for the GIF."""
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=252)
    frame = np.concatenate([pad(left), np.full((h, 6, 3), 252, np.uint8), pad(right)], axis=1)
    img = Image.fromarray(frame)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def save_gif(frames, out, colors=128, hold=15):
    """Write the frames as a looping GIF (10 fps), holding the last frame for a moment."""
    frames = frames + [frames[-1]] * hold
    pal = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)


class GifRecorder:
    """Per-test GIF: run_tests.run_episode calls it after every navigation step."""
    def __init__(self, env, target_height):
        """Add a visible copy of the goal to the 3D view; the map is built on the first frame."""
        vis = env.walk.pb.createVisualShape(p.GEOM_CYLINDER, radius=TARGET_RADIUS, length=target_height,
                                            rgbaColor=[*TARGET_RGB, 1])
        self.target_vis = env.walk.pb.createMultiBody(0, -1, vis, [100, 0, 0])
        self.target_height = target_height
        self.top = None                       # built on the first frame, once the episode has placed the map
        self.frames = []

    def __call__(self, env, rec, step, done, outcome):
        """Every 0.2 s of sim time: add one [map | 3D view] frame."""
        if self.top is None:
            self.top = TopDown(env)
        if step % 2 == 0 or done:                                      # every 0.2 s of sim time
            left = self.top.draw(rec["xy"], rec["yaw"][-1], step * 0.1, env._target_dist(), outcome if done else None)
            self.frames.append(side_by_side(left, chase_frame(env, self.target_vis, self.target_height)))

    def save(self, out):
        """Write this test's GIF."""
        if self.top is not None:
            self.top.close()
        save_gif(self.frames, out)
        return out


def all_tests_gif(R, out):
    """All tests on one top-down map, played at the same time."""
    from q2_navigation_env import GOAL
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.92)
    map_axes(ax, R[0]["layout"], GOAL, ARENA, REACH_DIST)
    lines = [ax.plot([], [], color=TEST_COLORS[r["k"]], lw=2.2, zorder=5)[0] for r in R]
    dots = [ax.scatter([], [], s=60, color=TEST_COLORS[r["k"]], edgecolor=INK, linewidth=1, zorder=7) for r in R]
    for r in R:
        ax.scatter(*r["xy"][0], s=55, color=SURFACE, edgecolor=TEST_COLORS[r["k"]], linewidth=2, zorder=6)
    title = ax.set_title("", loc="left", fontsize=10.5)
    T = max(len(r["t"]) for r in R)
    frames = []
    for i in list(range(0, T, 2)) + [T - 1]:
        done = 0
        for r, ln, dot in zip(R, lines, dots):
            j = min(i, len(r["t"]) - 1)
            ln.set_data(r["xy"][:j + 1, 0], r["xy"][:j + 1, 1])
            dot.set_offsets(r["xy"][j:j + 1])
            done += (j == len(r["t"]) - 1) and r["outcome"] == "success"
        title.set_text(f"all {len(R)} tests   t = {min(i, T - 1) * 0.1:4.1f} s   at goal: {done}/{len(R)}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    save_gif(frames, out, colors=64)
    return out


def main():
    """Command line: replay the 5 tests with rendering on and write all the GIFs."""
    import run_tests as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="final", choices=list(T.PRESETS))
    args = ap.parse_args()
    cfg = T.config_from_preset(args.preset)
    R = T.run_all(cfg, gifs=True)                     # replays the tests with the recorder on
    all_tests_gif(R, os.path.join(cfg["gif_dir"], "q2_all_tests.gif"))
    print("gifs:", ", ".join(sorted(os.listdir(cfg["gif_dir"]))))


if __name__ == "__main__":
    main()
