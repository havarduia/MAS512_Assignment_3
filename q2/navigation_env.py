"""Camera-based navigation for Lite3: walk to a red pillar without hitting obstacles.

Two levels:
  * low level  -- the trained omni walker (frozen PPO policy) turns a velocity command into
                  joint targets at 50 Hz (Lite3Env).
  * high level -- THIS env. Its policy sees only the onboard camera + body speeds and outputs a
                  velocity command (vx, vy, yaw rate) at 10 Hz.


Ground truth (robot-target distance, contacts) is used only for the reward and termination.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import torch
from stable_baselines3 import PPO
from walking_env import Lite3Env

torch.set_num_threads(1) 


ARENA = 4.0              # walled arena ARENA^2 (m)
SPAWN = 3.5              # robot start and target are placed within SPAWN^2
OUT_OF_BOUNDS = 5.0      # safety net the boundary walls normally stop the robot first
TARGET_RADIUS, TARGET_HEIGHT = 0.15, 0.8
TARGET_RGB = (0.9, 0.1, 0.1)
MIN_TARGET_DIST = 3.0    # target spawns at least this far from the robot
CLEARANCE = 0.6          # obstacles keep at least this gap from the robot start and the target

# obstacle pool: (kind, count). Sizes are randomised once per env when the pool is built
POOL = [("box", 8), ("pillar", 4), ("wall", 4), ("decoy", 4)]
MAX_OBSTACLES = sum(n for _, n in POOL)       # 20
DECOY_RGB = [(1.0, 0.55, 0.1), (0.2, 0.4, 0.9), (0.2, 0.7, 0.3), (0.95, 0.85, 0.2), (0.55, 0.25, 0.75)]
REACH_DIST = 0.5         # success when the robot base is this close to the target centre
SAFE_DIST = 0.35         # robot-obstacle gap (m) below which the "near" penalty starts

# The camera only looks forward, so navigation only moves where it can see: full speed ahead
# and turning, but just a creep backwards / sideways (blind reversing caused every collision).
NAV_RANGE = np.array([[-0.1, 1.2], # vx  (m/s)
                      [-0.2, 0.2], # vy  (m/s)
                      [-1.0, 1.0]])  # yaw (rad/s)

# timing
WALKER_STEPS = 5   # walker steps (0.02 s) per navigation step -> 10 Hz
MAX_NAV_STEPS = 600  # 60 s

#  camera (D435i mount) 
CAM_OFFSET = np.array([0.252, 0.0, 0.074])  # from the URDF, base frame
CAM_W, CAM_H = 64, 40
CAM_VFOV = 42.0   # deg; horizontal ~64 deg
NEAR, FAR = 0.1, 5.0
DEPTH_BINS = 16


class NavEnv(gym.Env):
    """Gymnasium env: walk to the red pillar using only the camera; the frozen walking policy does the walking."""
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self, walker_path, max_obstacles=MAX_OBSTACLES):
        """Load the walking env and walking policy, build the obstacle pool and camera; define observation / action spaces."""
        self.walk = Lite3Env(omni=False)   # commands are set by us, never resampled
        self.walker = PPO.load(walker_path, device="cpu")
        self.max_obstacles = max_obstacles #  train_nav.py raises this over time
        self._build_scene()

        # obs: depth bins (16), target (4), memory (2), body vel (3), previous action (3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (DEPTH_BINS + 12,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.proj = p.computeProjectionMatrixFOV(CAM_VFOV, CAM_W / CAM_H, NEAR, FAR)

    #world 
    def _build_scene(self):
        """Obstacles + target exist in the physics client (walker) and the render client (camera)."""
        self.cam = bullet_client.BulletClient(p.DIRECT)
        self.cam.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.cam.loadURDF("plane.urdf")
        rng = np.random.default_rng()

        # each obstacle: dict(kind, ids=(physics id, render id), half=(hx, hy, hz), round=bool)
        self.obstacles = []
        for kind, count in POOL:
            for i in range(count):
                if kind == "box":
                    half, rgb, round_ = [*rng.uniform(0.15, 0.45, 2), rng.uniform(0.3, 0.5)], \
                        [rng.uniform(0.45, 0.7)] * 3, False
                elif kind == "pillar":
                    r = rng.uniform(0.12, 0.35)
                    half, rgb, round_ = [r, r, rng.uniform(0.3, 0.5)], [0.5, 0.45, 0.4], True
                elif kind == "wall":
                    half, rgb, round_ = [rng.uniform(0.6, 1.5), 0.05, 0.4], [0.75, 0.72, 0.65], False
                else:  # decoy: coloured pillar that is NOT the target
                    half, rgb, round_ = [0.15, 0.15, 0.4], DECOY_RGB[i % len(DECOY_RGB)], True
                ids = tuple(self._make_body(client, half, rgb, round_) for client in (self.walk.pb, self.cam))
                self.obstacles.append(dict(kind=kind, ids=ids, half=half, round=round_))

        # boundary walls
        L = ARENA + 0.05
        self.boundary_ids = [] # physics-client ids, for collision checks
        for x, y, hx, hy in ((L, 0, 0.05, L), (-L, 0, 0.05, L), (0, L, L, 0.05), (0, -L, L, 0.05)):
            for client in (self.walk.pb, self.cam):
                body = self._make_body(client, [hx, hy, 0.4], [0.35, 0.35, 0.38], False)
                client.resetBasePositionAndOrientation(body, [x, y, 0.4], [0, 0, 0, 1])
                if client is self.walk.pb:
                    self.boundary_ids.append(body)

        # target: only rendered (walking into it is success, not a collision)
        vis = self.cam.createVisualShape(p.GEOM_CYLINDER, radius=TARGET_RADIUS, length=TARGET_HEIGHT,
                                         rgbaColor=[*TARGET_RGB, 1])
        self.target_id = self.cam.createMultiBody(0, -1, vis, [100, 0, 0])

    @staticmethod
    def _make_body(client, half, rgb, round_):
        """A static box or cylinder (obstacle or wall) in one PyBullet client."""
        if round_:
            col = client.createCollisionShape(p.GEOM_CYLINDER, radius=half[0], height=2 * half[2])
            vis = client.createVisualShape(p.GEOM_CYLINDER, radius=half[0], length=2 * half[2],
                                           rgbaColor=[*rgb, 1])
        else:
            col = client.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = client.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[*rgb, 1])
        return client.createMultiBody(0, col, vis, [100, 0, 0])

    def _place(self, ob, xy, yaw=0.0):
        """Move an obstacle to xy / yaw in both the physics client and the camera client."""
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        for client, body in zip((self.walk.pb, self.cam), ob["ids"]):
            client.resetBasePositionAndOrientation(body, [xy[0], xy[1], ob["half"][2]], quat)

    @staticmethod
    def _gap(point, ob, xy, yaw):
        """Distance from a point to the obstacle's footprint (circle or rotated rectangle)."""
        d = point - xy
        if ob["round"]:
            return np.linalg.norm(d) - ob["half"][0]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.abs([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - ob["half"][:2]
        return np.linalg.norm(np.maximum(local, 0)) + min(max(local[0], local[1]), 0)

    def _random_layout(self):
        """Start, target and a random subset of the pool placed with clearance around both."""
        rng = self.np_random
        start = rng.uniform(-SPAWN, SPAWN, 2)
        while True:
            target = rng.uniform(-SPAWN, SPAWN, 2)
            if np.linalg.norm(target - start) >= MIN_TARGET_DIST:
                break
        n = rng.integers(0, self.max_obstacles + 1)
        chosen = rng.permutation(len(self.obstacles))[:n]
        layout = []  # (obstacle index, xy, yaw)
        for i in chosen:
            ob = self.obstacles[i]
            for _ in range(50):  # rejection sampling
                xy, yaw = rng.uniform(-ARENA + 0.3, ARENA - 0.3, 2), rng.uniform(0, np.pi)
                if (self._gap(start, ob, xy, yaw) > CLEARANCE and self._gap(target, ob, xy, yaw) > CLEARANCE
                        and all(np.linalg.norm(xy - o[1]) > 0.5 for o in layout)):
                    layout.append((i, xy, yaw))
                    break
        return start, target, layout

    # perception 
    def _camera(self):
        """Render RGB + metric depth from the head camera."""
        pos, quat = self.walk.pb.getBasePositionAndOrientation(self.walk.robot)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        eye = np.array(pos) + R @ CAM_OFFSET
        view = p.computeViewMatrix(eye, eye + R[:, 0], R[:, 2])
        _, _, rgb, depth, _ = self.cam.getCameraImage(CAM_W, CAM_H, view, self.proj,
                                                      renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (CAM_H, CAM_W, 4))[:, :, :3].astype(np.float32)
        buf = np.reshape(depth, (CAM_H, CAM_W))
        depth = FAR * NEAR / (FAR - (FAR - NEAR) * buf)   # OpenGL depth buffer -> metres
        return rgb, depth

    def _perceive(self):
        """Camera image -> 16 obstacle depth bins, goal (visible, bearing, size, distance) and goal memory."""
        rgb, depth = self._camera()

        # target: clearly red pixels
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (r > 80) & (r > 2.0 * g) & (r > 2.0 * b)

        # obstacles: nearest thing in each column band, using only rows at/above the horizon
        # (the target is not an obstacle, so its pixels are ignored here)
        upper = np.where(mask, FAR, depth)[: CAM_H // 2 + 1]
        bins = upper.reshape(upper.shape[0], DEPTH_BINS, -1).min(axis=(0, 2)) / FAR   # 0..1

        visible = mask.sum() >= 2
        if visible:
            cols = np.nonzero(mask)[1]
            bearing = 1.0 - 2.0 * (cols.mean() + 0.5) / CAM_W      # +1 = left edge, -1 = right edge
            size = np.sqrt(mask.mean())
            dist = np.median(depth[mask]) / FAR
            self.last_bearing, self.time_unseen = bearing, 0.0
        else:
            bearing = size = dist = 0.0
            self.time_unseen = min(self.time_unseen + WALKER_STEPS * self.walk.dt, 10.0)
        target = np.array([float(visible), bearing, size, dist])
        memory = np.array([self.last_bearing, self.time_unseen / 10.0])
        return bins, target, memory

    def _obs(self):
        """The 28-number observation: depth bins, goal, memory, body velocity, previous action."""
        bins, target, memory = self._perceive()
        self.target_visible = target[0] > 0
        _, lin, ang, _ = self.walk._base_state()
        body = np.array([lin[0], lin[1], ang[2]])
        return np.concatenate([bins, target, memory, body, self.last_action]).astype(np.float32)

    #  ground truth (reward only) 
    def _robot_xy(self):
        """Robot body position (x, y)."""
        return np.array(self.walk.pb.getBasePositionAndOrientation(self.walk.robot)[0][:2])

    def _target_dist(self):
        """Distance from the robot body to the goal (m)."""
        return np.linalg.norm(self._robot_xy() - self.target_xy)

    def _solid_ids(self):
        """Everything the robot must not touch: the obstacles and the arena walls."""
        return [ob["ids"][0] for ob in self.obstacles] + self.boundary_ids

    def _hit_obstacle(self):
        """True if any part of the robot touches an obstacle or wall."""
        return any(self.walk.pb.getContactPoints(self.walk.robot, b) for b in self._solid_ids())

    def _clearance(self):
        """Smallest gap between any robot link and any obstacle, capped at SAFE_DIST."""
        gaps = [pt[8] for b in self._solid_ids()
                for pt in self.walk.pb.getClosestPoints(self.walk.robot, b, distance=SAFE_DIST)]
        return min(gaps, default=SAFE_DIST)

    # gym API 
    def reset(self, seed=None, options=None):
        """New episode: random layout, place obstacles, goal and robot, return the first observation."""
        super().reset(seed=seed)
        start, self.target_xy, layout = self._random_layout()
        for k, ob in enumerate(self.obstacles): # park everything far away first
            self._place(ob, (100 + 3 * k, 0))
        for i, xy, yaw in layout:
            self._place(self.obstacles[i], xy, yaw)
        self.layout = layout   # kept for plotting the map
        self.cam.resetBasePositionAndOrientation(self.target_id, [*self.target_xy, TARGET_HEIGHT / 2],
                                                 [0, 0, 0, 1])

        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.walker_obs, _ = self.walk.reset(seed=int(self.np_random.integers(2**31)),
                                             options={"pose": (*start, yaw)})
        self.walk.command = np.zeros(3)
        self.walker_obs = self.walk._obs()
        self.last_action = np.zeros(3)
        self.last_bearing, self.time_unseen = 0.0, 10.0
        self.steps = 0
        self.prev_dist = self._target_dist()
        return self._obs(), {}

    def step(self, action):
        """Navigation action -> velocity command -> 5 walker steps (0.1 s) -> observation, reward, done."""
        action = np.clip(action, -1, 1)
        # [-1, 1] -> velocity command within NAV_RANGE
        lo, hi = NAV_RANGE[:, 0], NAV_RANGE[:, 1]
        self.walk.command = np.where(action >= 0, action * hi, -action * lo)

        fell = hit = False
        for _ in range(WALKER_STEPS):
            a, _ = self.walker.predict(self.walker_obs, deterministic=True)
            self.walker_obs, _, fell, _, _ = self.walk.step(a)
            hit = hit or self._hit_obstacle()
            if fell or hit:
                break

        obs = self._obs()
        self.steps += 1
        dist = self._target_dist()
        reached = dist < REACH_DIST
        lost = np.max(np.abs(self._robot_xy())) > OUT_OF_BOUNDS

        terms = {
            "progress": 5.0 * (self.prev_dist - dist), # +5 per metre closer
            "time":    -0.02,
            "unseen":  -0.02 * (not self.target_visible),  # keep the target in view when possible
            "smooth":  -0.02 * np.sum((action - self.last_action) ** 2),
            # keep a margin: 0 when >= SAFE_DIST away, -1 when touching
            "near":    -1.0 * max(SAFE_DIST - self._clearance(), 0.0) / SAFE_DIST,
        }
        if reached:
            terms["reached"] = 10.0
        if hit:
            terms["crash"] = -10.0
        if fell:
            terms["fell"] = -5.0
        if lost:
            terms["lost"] = -5.0
        self.prev_dist = dist
        self.last_action = action

        terminated = bool(reached or hit or fell or lost)
        truncated = self.steps >= MAX_NAV_STEPS
        info = dict(terms, success=reached, collision=hit, fell=fell)
        return obs, float(sum(terms.values())), terminated, truncated, info

    def close(self):
        """Shut down both PyBullet clients."""
        self.walk.close()
        self.cam.disconnect()


if __name__ == "__main__":
    import sys, time
    env = NavEnv(sys.argv[1] if len(sys.argv) > 1 else "models/walking_policy.zip")
    obs, _ = env.reset(seed=0)
    print("obs", np.round(obs, 2))
    t = time.time()
    n = 0
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        n += 1
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()
    print(f"{n / (time.time() - t):.0f} nav steps/s per env")
    env.close()
    print("ok")
