#Navigation env for Q2: NavEnv (navigation_env.py) with a goal that can be seen from across the arena.

import os, sys
import numpy as np
import pybullet as p
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the local copies in q2/
import navigation_env                               
from navigation_env import (NavEnv, TARGET_RADIUS, TARGET_RGB, CAM_W, CAM_H, CAM_VFOV, CAM_OFFSET,  
                     NEAR, FAR, SPAWN, CLEARANCE, MIN_TARGET_DIST)

POOL_SEED = 7
DEPTH_RANGE = (0.28, 20.0)      # depth sensing range (m): D435i min-Z max set to 20 m (spec max 10 m)
# the predefined Q2 maps 
# (pool index, centre xy, yaw). Pool: 0-7 boxes, 8-11 pillars, 12-15 walls, 16-19 coloured decoys
GOAL = np.array([3.0, 2.8])
TESTS = [((-3.2, -3.2), 45), ((-3.3, -1.0), 0), ((-3.3, 1.2), 0), ((-1.5, -3.3), 90), ((0.8, -3.3), 90)]
MAPS = {
    # the Q2 test: clutter to steer around on every route
    "standard": [(14, (-1.4, -1.5), np.pi / 2),# wall across the lower-left routes
                 (15, (1.9, -0.6), np.pi / 4), # diagonal wall, lower right
                 (12, (0.4, 3.3), 0.0), # wall along the top
                 (2, (-2.5, -1.8), 0.5), (4, (0.0, -1.2), 0.3), (0, (-0.7, 1.7), 0.0),
                 (7, (1.2, 1.2), 0.0), (5, (2.5, 1.0), 0.0),
                 (11, (-2.2, 0.6), 0.0), (9, (1.9, 2.3), 0.0), (10, (0.4, -0.1), 0.0),
                 (16, (3.0, -2.5), 0.0), (17, (-1.2, 3.0), 0.0)],
    # stress test: walls / boxes hide the goal from every start
    "hidden_goal": [(13, (-1.2, 0.0), np.pi / 2), (15, (1.3, 1.2), np.pi / 4), (12, (1.9, -1.4), 0.0),
                    (4, (0.2, -1.6), 0.3), (0, (-0.2, 1.9), 0.0), (2, (-2.4, -2.0), 0.5),
                    (7, (2.6, 0.6), 0.0), (5, (-0.6, -2.9), 0.0),
                    (11, (0.0, 0.2), 0.0), (9, (2.0, 2.0), 0.0), (10, (-2.4, 2.6), 0.0),
                    (16, (1.0, 3.2), 0.0), (17, (3.3, -2.6), 0.0)],
}


class Q2NavigationEnv(NavEnv):
    """NavEnv with the Q2 settings (see the top of this file); everything else is inherited from NavEnv."""
    def __init__(self, walker_path, max_obstacles=20, target_height=4.0, render_far=20.0, depth_range=DEPTH_RANGE,
                 fixed_map_frac=0.0, train_map="standard", crash_penalty=-30.0, near_scale=2.0):
        """Build the normal NavEnv (with the seeded obstacle pool), then apply the Q2 goal / camera / penalty settings."""
        rng = np.random.default_rng  # seeded obstacle pool
        navigation_env.np.random.default_rng = lambda *a: rng(POOL_SEED)
        try:
            super().__init__(walker_path, max_obstacles=max_obstacles)
        finally:
            navigation_env.np.random.default_rng = rng
        self.target_height, self.render_far = target_height, render_far
        self.depth_min, self.depth_max = depth_range or (NEAR, FAR)
        self.fixed_map_frac, self.train_map = fixed_map_frac, train_map
        self.crash_penalty, self.near_scale = crash_penalty, near_scale
        self.cam.removeBody(self.target_id)  # rebuild the goal at the new height
        vis = self.cam.createVisualShape(p.GEOM_CYLINDER, radius=TARGET_RADIUS, length=target_height,
                                         rgbaColor=[*TARGET_RGB, 1])
        self.target_id = self.cam.createMultiBody(0, -1, vis, [100, 0, 0])
        if render_far:
            self.proj = p.computeProjectionMatrixFOV(CAM_VFOV, CAM_W / CAM_H, NEAR, render_far)

    def _camera(self):
        """Robot camera image: colour rendered out to render_far, depth clipped to depth_range (metres)."""
        far = self.render_far or FAR
        pos, quat = self.walk.pb.getBasePositionAndOrientation(self.walk.robot)
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        eye = np.array(pos) + R @ CAM_OFFSET
        view = p.computeViewMatrix(eye, eye + R[:, 0], R[:, 2])
        _, _, rgb, depth, _ = self.cam.getCameraImage(CAM_W, CAM_H, view, self.proj, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (CAM_H, CAM_W, 4))[:, :, :3].astype(np.float32)
        buf = np.reshape(depth, (CAM_H, CAM_W))
        depth = far * NEAR / (far - (far - NEAR) * buf)
        return rgb, np.clip(depth, self.depth_min, self.depth_max)

    def _perceive(self):
        """navigation_env.NavEnv._perceive with the depth scale of this camera (depth_max instead of 5 m)."""
        rgb, depth = self._camera()
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (r > 80) & (r > 2.0 * g) & (r > 2.0 * b)  # red goal pixels
        upper = np.where(mask, self.depth_max, depth)[: CAM_H // 2 + 1]  # rows at / above the horizon
        bins = upper.reshape(upper.shape[0], navigation_env.DEPTH_BINS, -1).min(axis=(0, 2)) / self.depth_max
        visible = mask.sum() >= 2
        if visible:
            cols = np.nonzero(mask)[1]
            bearing = 1.0 - 2.0 * (cols.mean() + 0.5) / CAM_W
            size = np.sqrt(mask.mean())
            dist = np.median(depth[mask]) / self.depth_max
            self.last_bearing, self.time_unseen = bearing, 0.0
        else:
            bearing = size = dist = 0.0
            self.time_unseen = min(self.time_unseen + navigation_env.WALKER_STEPS * self.walk.dt, 10.0)
        target = np.array([float(visible), bearing, size, dist])
        memory = np.array([self.last_bearing, self.time_unseen / 10.0])
        return bins, target, memory

    def fixed_layout(self, name):
        """Obstacle placements of a predefined map: [(pool index, xy, yaw), ...]."""
        return [(i, np.array(xy), yaw) for i, xy, yaw in MAPS[name]]

    def _random_layout(self):
        """Start, goal and obstacles for a new episode: the Q2 map (fixed_map_frac of the time) or a random map."""
        rng = self.np_random
        if rng.uniform() >= self.fixed_map_frac:
            return super()._random_layout()
        layout = self.fixed_layout(self.train_map)
        while True:   # random free start, > 3 m from the goal
            start = rng.uniform(-SPAWN, SPAWN, 2)
            if (np.linalg.norm(start - GOAL) >= MIN_TARGET_DIST and all(
                    self._gap(start, self.obstacles[i], xy, yaw) > CLEARANCE for i, xy, yaw in layout)):
                return start, GOAL.copy(), layout

    def reset(self, seed=None, options=None):
        """New episode, then move the (taller) goal pillar to the goal and return the first observation."""
        super().reset(seed=seed, options=options)
        self.cam.resetBasePositionAndOrientation(self.target_id, [*self.target_xy, self.target_height / 2],
                                                 [0, 0, 0, 1])
        return self._obs(), {}

    def step(self, action):
        """One 0.1 s navigation step (NavEnv.step), with the bigger crash and near-obstacle penalties."""
        obs, r, term, trunc, info = super().step(action)
        extra = (self.near_scale - 1.0) * info["near"] # scale the margin penalty
        if info["collision"]:
            extra += self.crash_penalty - info["crash"]  # -10 -> crash_penalty
        info["near"] *= self.near_scale
        if info["collision"]:
            info["crash"] = self.crash_penalty
        return obs, r + extra, term, trunc, info
