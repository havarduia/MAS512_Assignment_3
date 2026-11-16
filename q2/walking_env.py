"""Lite3 quadruped walking environment (PyBullet + Gymnasium).

The policy outputs 12 joint-angle offsets from DEFAULT_POSE; a PD controller turns them into
torques. The robot is asked to follow a body-velocity command (vx, vy, yaw rate) with a trot:
diagonal leg pairs (FL+HR, FR+HL) step together, timed by a gait clock that is part of the
observation.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

#robot URDF and joint names
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Lite3/urdf/Lite3.urdf")
LEGS = ["FL", "FR", "HL", "HR"]
MOTORS = [f"{leg}_{j}" for leg in LEGS for j in ("HipX_joint", "HipY_joint", "Knee_joint")]
DEFAULT_POSE = np.array([0.0, -0.8, 1.6] * 4)  # HipX, HipY, Knee per leg (rad)

# initial condition: the robot starts at the origin, facing +x, with a random gait phase
BASE_HEIGHT = 0.29 # target body height (m), the standing height the self-check prints
MIN_HEIGHT = 0.22  # episode ends if the body sags below this
FOOT_Z_STAND = 0.022  # foot-link height when the foot rests on the ground
LEVEL_TOL = np.sin(np.radians(2.0))   # tilt scale for the "level" reward

# gait
GAIT_PERIOD = 0.4 # s for one full trot cycle up to 1 m/s: FL+HR swing for half, then FR+HL
FAST_PERIOD = 0.3  # s per cycle at FAST_SPEED; quicker steps make high speed possible
FAST_SPEED = 2.0 # m/s
SWING_HEIGHT = 0.08 # m, how high a swinging foot should lift
STAND_CMD = 0.1 # commands smaller than this mean "stand still" (all feet down)

#commands
VX_RANGE = (0.4, 1.2)  # forward-only mode (m/s)
OMNI_RANGE = np.array([[-0.6, 2.0], # vx  (m/s)
                       [-0.5, 0.5],  # vy  (m/s)
                       [-1.0, 1.0]]) # yaw (rad/s)
RESAMPLE_S = (3.0, 5.0)   # omni mode: new command every 3-5 s

#Rough ground training
ROUGH_TILE = 0.25  # m
ROUGH_WINDOW = 24 # tiles per side of the window (6 m)
RECENTRE = 1.0  # m off-centre before the window moves
ROUGH_MAX = 0.08 # m, tallest tile at difficulty 1
TILE_HALF_Z = 0.1 # tiles are 20 cm thick boxes; hidden ones sit just under the slab
PUSH_EVERY = (2.0, 4.0)   # s between random pushes
PUSH_MAX = 1.0  # m/s horizontal velocity kick at difficulty 1
FRICTION_RANGE = (0.3, 1.2)
PAYLOAD_RANGE = (-1.0, 3.0)  # kg added to the torso
SLOPE_MAX = 10.0    #  30% of episodes tilt gravity (= walking on a slope)
KP_RANGE = (0.9, 1.1)  # motor strength scale


class Lite3Env(gym.Env):
    """Gymnasium env: the Lite3 quadruped in PyBullet, learning to follow velocity commands."""
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, omni=False, rough=False, sim_dt=0.005, decimation=4, max_steps=1000,
                 kp=45.0, kd=1.0, action_scale=0.5):
        """PyBullet setup: ground, robot URDF, joint limits, friction; observation (50) and action (12) spaces."""
        self.omni, self.rough = omni, rough
        self.difficulty = 1.0          # rough mode: 0..1, raised by train_walking.py over training
        self.sim_dt, self.decimation, self.max_steps = sim_dt, decimation, max_steps
        self.kp, self.kd, self.action_scale = kp, kd, action_scale
        self.dt = sim_dt * decimation  # policy time step (0.02 s = 50 Hz)

        # headless: no screen on coder.
        self.pb = bullet_client.BulletClient(p.DIRECT)
        self.pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pb.setGravity(0, 0, -9.81)
        self.pb.setTimeStep(sim_dt)
        self.plane = self.pb.loadURDF("plane.urdf")
        self.robot = self.pb.loadURDF(URDF, [0, 0, 0.35], flags=p.URDF_USE_INERTIA_FROM_FILE)

        names = {self.pb.getJointInfo(self.robot, j)[1].decode(): j
                 for j in range(self.pb.getNumJoints(self.robot))}
        self.motor_ids = [names[m] for m in MOTORS]
        self.foot_ids = [names[f"{leg}_Ankle"] for leg in LEGS]   # FL, FR, HL, HR
        self.foot_set = set(self.foot_ids)
        info = [self.pb.getJointInfo(self.robot, j) for j in self.motor_ids]
        self.q_low = np.array([i[8] for i in info])
        self.q_high = np.array([i[9] for i in info])
        self.tau_max = np.array([i[10] for i in info])

        # sensor links have no inertial in the URDF -> PyBullet gives each 1 kg. Make them massless.
        for j in names.values():
            if self.pb.getJointInfo(self.robot, j)[12].decode() in ("D435I_LINK", "MID360_LINK", "IMU_LINK"):
                self.pb.changeDynamics(self.robot, j, mass=1e-3, localInertiaDiagonal=[1e-6] * 3)
        for j in self.foot_ids:
            self.pb.changeDynamics(self.robot, j, lateralFriction=0.8)
        self.pb.changeDynamics(self.plane, -1, lateralFriction=0.8)
        # turn off PyBullet's built-in velocity motors so our torques are the only actuation
        self.pb.setJointMotorControlArray(self.robot, self.motor_ids, p.VELOCITY_CONTROL,
                                          forces=np.zeros(12))
        self.base_mass = self.pb.getDynamicsInfo(self.robot, -1)[0]
        self.kp_scale = 1.0
        self.tile_top = None # rough mode: tile top heights
        if rough:
            self._build_tiles()

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (50,), np.float32)

    # sensors
    def _joint_state(self):
        """Angles and velocities of the 12 joints."""
        s = self.pb.getJointStates(self.robot, self.motor_ids)
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

    def _base_state(self):
        """Position (world), and lin vel / ang vel / gravity direction in the body frame."""
        pos, quat = self.pb.getBasePositionAndOrientation(self.robot)
        lin, ang = self.pb.getBaseVelocity(self.robot)
        R = np.array(self.pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return np.array(pos), R.T @ lin, R.T @ ang, R.T @ np.array([0, 0, -1.0])

    def _ground_links(self):
        """Robot links touching the ground. The ground is self.ground: the plane, or the set of
        bodies a test terrain is built from (terrain.py)."""
        ground = getattr(self, "ground", None) or {self.plane}
        return {c[3] for c in self.pb.getContactPoints(self.robot) if c[2] in ground}

    def _foot_contacts(self):
        """Which of the 4 feet touch the ground (FL, FR, HL, HR)."""
        touching = self._ground_links()
        return np.array([f in touching for f in self.foot_ids])

    def _foot_velocities(self):
        """World velocity of each foot (used for the slip penalty)."""
        return np.array([self.pb.getLinkState(self.robot, f, computeLinkVelocity=1)[6]
                         for f in self.foot_ids])

    def _foot_heights(self):
        """Foot height above the ground directly under each foot."""
        pts = [self.pb.getLinkState(self.robot, f)[0] for f in self.foot_ids]
        return np.array([z - self._ground_z(x, y) for x, y, z in pts])

    # terrain
    def _build_tiles(self):
        """Big slab (top at z = 0) + a window of tiles that follows the robot."""
        pb = self.pb
        pb.removeBody(self.plane)
        slab = pb.createMultiBody(0, pb.createCollisionShape(p.GEOM_BOX, halfExtents=[60, 60, 0.5]),
                                  pb.createVisualShape(p.GEOM_BOX, halfExtents=[60, 60, 0.5],
                                                       rgbaColor=[0.8, 0.78, 0.72, 1]), [0, 0, -0.5])
        hx = [ROUGH_TILE / 2, ROUGH_TILE / 2, TILE_HALF_Z]
        n = ROUGH_WINDOW * ROUGH_WINDOW
        ids = pb.createMultiBody(0, pb.createCollisionShape(p.GEOM_BOX, halfExtents=hx),
                                 pb.createVisualShape(p.GEOM_BOX, halfExtents=hx, rgbaColor=[0.62, 0.6, 0.55, 1]),
                                 batchPositions=[[0, 0, -1.0]] * n)
        self.tiles = list(ids)
        self.plane = slab
        self.ground = {slab, *self.tiles}
        self.terrain = dict(kind="flat", top=0.0, seed=0.0, spawn=(0.0, 0.0))
        self.window = None   # global cell index of the window corner
        self.tile_top = True # marks rough mode for _ground_z

    def _cell_height(self, I, J):
        """Tile top height for global cell indices I, J (arrays ok); <= 0 means no tile (flat)."""
        t = self.terrain
        if t["kind"] == "flat":
            h = np.full(np.broadcast(I, J).shape, -1.0)
        else:
            if t["kind"] == "steps":  # 1 m blocks of equal height
                block = int(1.0 / ROUGH_TILE)
                I, J = np.floor_divide(I, block), np.floor_divide(J, block)
            u = np.sin(I * 12.9898 + J * 78.233 + t["seed"]) * 43758.5453
            h = (u - np.floor(u)) * t["top"]  # hash -> uniform [0, top)
        return h

    def _heights_at(self, I, J):
        """Tile heights, keeping a flat pad around the spawn point."""
        h = self._cell_height(I, J)
        sx, sy = self.terrain["spawn"]
        x, y = (np.asarray(I) + 0.5) * ROUGH_TILE, (np.asarray(J) + 0.5) * ROUGH_TILE
        return np.where((np.abs(x - sx) < 0.6) & (np.abs(y - sy) < 0.6), -1.0, h)

    def _ground_z(self, x, y):
        """Ground height at (x, y); 0 on flat ground."""
        if self.tile_top is None:
            return 0.0
        I, J = int(np.floor(x / ROUGH_TILE)), int(np.floor(y / ROUGH_TILE))
        return max(float(self._heights_at(I, J)), 0.0)

    def _base_ground_z(self, x, y):
        """Ground height used for the body-height reward / sag check (stairs/ overrides this with
        the mean over the body's footprint)."""
        return self._ground_z(x, y)

    def _place_window(self, x, y):
        """Put the tile window around (x, y) and set every tile to its cell's height."""
        I0 = int(np.floor(x / ROUGH_TILE)) - ROUGH_WINDOW // 2
        J0 = int(np.floor(y / ROUGH_TILE)) - ROUGH_WINDOW // 2
        self.window = (I0, J0)
        I, J = np.meshgrid(np.arange(I0, I0 + ROUGH_WINDOW), np.arange(J0, J0 + ROUGH_WINDOW), indexing="ij")
        h = self._heights_at(I, J).ravel()
        for body, i, j, z in zip(self.tiles, I.ravel(), J.ravel(), h):
            zc = (z if z > 0 else -0.01) - TILE_HALF_Z
            self.pb.resetBasePositionAndOrientation(body, [(i + 0.5) * ROUGH_TILE, (j + 0.5) * ROUGH_TILE, zc],
                                                    [0, 0, 0, 1])

    def _follow_robot(self, x, y):
        """Rough mode: move the tile window when the robot gets near its edge."""
        cx = (self.window[0] + ROUGH_WINDOW / 2) * ROUGH_TILE
        cy = (self.window[1] + ROUGH_WINDOW / 2) * ROUGH_TILE
        if max(abs(x - cx), abs(y - cy)) > RECENTRE:
            self._place_window(x, y)

    def _shape_terrain(self, spawn):
        """New random terrain: flat (20%), rough tiles (50%) or 1 m steps (30%)."""
        rng = self.np_random
        kind = rng.uniform()
        self.terrain = dict(kind="flat" if kind < 0.2 else "rough" if kind < 0.7 else "steps",
                            top=ROUGH_MAX * self.difficulty * rng.uniform(0.3, 1.0),
                            seed=rng.uniform(0, 1000), spawn=tuple(spawn))
        self._place_window(*spawn)

    def _randomise_physics(self):
        """Rough mode: random friction, payload, motor strength and slope for this episode."""
        rng = self.np_random
        mu = rng.uniform(*FRICTION_RANGE)
        for g in self.ground:
            self.pb.changeDynamics(g, -1, lateralFriction=mu)
        for f in self.foot_ids:
            self.pb.changeDynamics(self.robot, f, lateralFriction=mu)
        self.pb.changeDynamics(self.robot, -1, mass=self.base_mass + rng.uniform(*PAYLOAD_RANGE))
        self.kp_scale = rng.uniform(*KP_RANGE)
        g = np.array([0, 0, -9.81])
        if rng.uniform() < 0.3:
            a, az = np.radians(rng.uniform(0, SLOPE_MAX * self.difficulty)), rng.uniform(0, 2 * np.pi)
            g = 9.81 * np.array([np.sin(a) * np.cos(az), np.sin(a) * np.sin(az), -np.cos(a)])
        self.pb.setGravity(*g)

    def _schedule_push(self):
        """Rough mode: choose when the next random push happens."""
        self.next_push = self.steps + int(self.np_random.uniform(*PUSH_EVERY) / self.dt)

    def _maybe_push(self):
        """Rough mode: kick the body sideways when a push is due."""
        if self.steps >= self.next_push:
            dv = self.np_random.uniform(-1, 1, 2) * PUSH_MAX * self.difficulty
            lin, ang = self.pb.getBaseVelocity(self.robot)
            self.pb.resetBaseVelocity(self.robot, [lin[0] + dv[0], lin[1] + dv[1], lin[2]], ang)
            self._schedule_push()

    def _body_contacts(self):
        """Robot links other than feet touching the ground; -1 = torso."""
        return self._ground_links() - self.foot_set

    def _obs(self):
        """The 50-number observation (listed at the top of this file)."""
        _, lin, ang, grav = self._base_state()
        q, dq = self._joint_state()
        if self._standing():
            clock = [0.0, 0.0]   # no gait while standing
        else:
            clock = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
        return np.concatenate([lin, ang, grav, self.command, q - DEFAULT_POSE,
                               dq * 0.05, self.last_action, clock]).astype(np.float32)

    #commands and gait clock
    def _standing(self):
        """True when the command is (almost) zero: stand still, no gait."""
        return np.linalg.norm(self.command) < STAND_CMD

    def _gait_period(self):
        """Shorter cycle (quicker steps) above 1 m/s forward."""
        return np.interp(abs(self.command[0]), [1.0, FAST_SPEED], [GAIT_PERIOD, FAST_PERIOD])

    def _desired_stance(self):
        """Which feet should be on the ground now: all when standing, else trot by the clock."""
        if self._standing():
            return np.ones(4, dtype=bool)
        first_half = self.phase < 0.5   # first half FL+HR down, second half FR+HL down
        return np.array([first_half, not first_half, not first_half, first_half])

    def _sample_command(self):
        """A new random velocity command (forward-only or omni; 20% stand still in omni)."""
        rng = self.np_random
        if not self.omni:
            return np.array([rng.uniform(*VX_RANGE), 0.0, 0.0])
        kind = rng.uniform()
        if kind < 0.20:   # stand still
            return np.zeros(3)
        cmd = rng.uniform(OMNI_RANGE[:, 0], OMNI_RANGE[:, 1])
        if kind < 0.35:  # one axis only (pure side step, turn, ...)
            keep = rng.integers(3)
            cmd = np.where(np.arange(3) == keep, cmd, 0.0)
        if np.linalg.norm(cmd) < STAND_CMD:  # too small to be a real move
            cmd[:] = 0.0
        return cmd

    def _schedule_resample(self):
        """Omni mode: choose when the command changes next (every 3-5 s)."""
        self.next_resample = self.steps + int(self.np_random.uniform(*RESAMPLE_S) / self.dt)

    #  gym API
    def reset(self, seed=None, options=None):
        """options={"pose": (x, y, yaw)} sets the start pose (default origin, facing +x)."""
        super().reset(seed=seed)
        default = (0.0, 0.0, 0.0)
        if self.rough:   # random start somewhere on the field
            default = (0.0, 0.0, self.np_random.uniform(-np.pi, np.pi))
        x, y, yaw = (options or {}).get("pose", default)
        if self.rough:
            self._shape_terrain((x, y))
            self._randomise_physics()
        self.pb.resetBasePositionAndOrientation(self.robot, [x, y, self._ground_z(x, y) + 0.30],
                                                self.pb.getQuaternionFromEuler([0, 0, yaw]))
        self.pb.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        for j, q in zip(self.motor_ids, DEFAULT_POSE):
            self.pb.resetJointState(self.robot, j, q, 0.0)
        self.steps = 0
        self.command = self._sample_command()   # desired body velocity (vx, vy, yaw rate)
        self._schedule_resample()
        self.phase = self.np_random.uniform(0, 1)
        self.last_action = np.zeros(12)
        self.prev_action = np.zeros(12)
        self._schedule_push()
        return self._obs(), {}

    def step(self, action):
        """Apply the action (joint targets, PD control for 4 physics steps), then compute reward and done."""
        #  act: PD control at sim rate, policy at sim rate / decimation 
        action = np.clip(action, -1, 1)
        q_target = np.clip(DEFAULT_POSE + self.action_scale * action, self.q_low, self.q_high)
        for _ in range(self.decimation):
            q, dq = self._joint_state()
            tau = np.clip(self.kp_scale * self.kp * (q_target - q) - self.kd * dq,
                          -self.tau_max, self.tau_max)
            self.pb.setJointMotorControlArray(self.robot, self.motor_ids, p.TORQUE_CONTROL, forces=tau)
            self.pb.stepSimulation()

        # reward 
        pos, lin, ang, grav = self._base_state()
        pos = pos.copy()
        pos[2] -= self._base_ground_z(pos[0], pos[1])  # height above the ground under the robot
        q, dq = self._joint_state()
        contact = self._foot_contacts()
        body = self._body_contacts()  # knees / shins / thighs / torso on the ground
        terms = self._reward_terms(pos, lin, ang, grav, q, dq, tau, action, contact)
        terms["body_contact"] = -2.0 * len(body)

        fallen = (pos[2] < MIN_HEIGHT # sagged too low
                  or grav[2] > -0.5 # tilted > ~60 deg
                  or -1 in body) # torso hit the ground
        if fallen:
            terms["fall"] = -10.0

        # advance time 
        self.prev_action = self.last_action
        self.last_action = action
        self.steps += 1
        if self._standing():
            self.phase = 0.0   # frozen; walking restarts at the start of a cycle
        else:
            self.phase = (self.phase + self.dt / self._gait_period()) % 1.0
        if self.omni and self.steps >= self.next_resample:
            self.command = self._sample_command()
            self._schedule_resample()

        if self.rough:
            self._maybe_push()
            self._follow_robot(pos[0], pos[1])
        truncated = self.steps >= self.max_steps
        return self._obs(), float(sum(terms.values())), bool(fallen), truncated, terms

    def _reward_terms(self, pos, lin, ang, grav, q, dq, tau, action, contact):
        """Per-step reward, split into named terms (eval.py prints their averages)."""
        cmd = self.command

        # gait: feet should follow the clock, and swinging feet should lift
        stance = self._desired_stance()
        gait = np.mean(contact == stance)                        # 1 = every foot as the clock says
        lift = np.clip(self._foot_heights() - FOOT_Z_STAND, 0, SWING_HEIGHT) / SWING_HEIGHT
        clearance = np.sum(lift * ~stance) / 2                   # 0..1, 0 when standing

        foot_v = self._foot_velocities()
        slip = np.sum(np.sum(foot_v[:, :2] ** 2, axis=1) * contact)   # planted feet sliding
        airborne = float(contact.sum() < 2)                            # jumping / flight
        dpose = q - DEFAULT_POSE
        lin_err = np.sum((cmd[:2] - lin[:2]) ** 2)
        yaw_err = (cmd[2] - ang[2]) ** 2

        terms = {
            "alive":       1.5,   # staying up must always beat falling, even while still clumsy
            "track_lin":   3.0 * np.exp(-lin_err / 0.25),
            "track_lin_fine": 1.0 * np.exp(-lin_err / 0.02),
            "track_yaw":   1.0 * np.exp(-yaw_err / 0.1),
            "track_yaw_fine": 0.5 * np.exp(-yaw_err / 0.02),
            "gait":        1.5 * gait,
            "clearance":   0.5 * clearance,
            "height":      0.5 * np.exp(-((pos[2] - BASE_HEIGHT) / 0.03) ** 2),
            #  stand straight 
            # level: sharp bonus for a flat back (1 at 0 deg, 0.75 at 1 deg, ~0 above 3 deg);
            # the quadratic penalty below only matters for big tilts
            "level":       1.0 * np.exp(-np.sum(grav[:2] ** 2) / LEVEL_TOL ** 2),
            "orientation": -10.0 * np.sum(grav[:2] ** 2),   # pitch / roll (rear sag shows up here)
            "lin_z":       -2.0 * lin[2] ** 2,
            "ang_xy":      -0.05 * np.sum(ang[:2] ** 2),
            "hip_pose":    -1.0 * np.sum(dpose[0::3] ** 2),  # legs under the body, not splayed
            "leg_pose":    -0.2 * np.sum(dpose[1::3] ** 2 + dpose[2::3] ** 2),  # front & rear alike
            # smooth, no jiggle 
            "action_rate": -0.1 * np.sum((action - self.last_action) ** 2),
            # second difference: punishes back-and-forth jitter, not the smooth swing of the gait
            "action_smooth": -1.0 * np.sum((action - 2 * self.last_action + self.prev_action) ** 2),
            "joint_vel":   -5e-4 * np.sum(dq ** 2),
            "torque":      -1e-4 * np.sum(tau ** 2),
            "slip":        -0.1 * slip,
            "airborne":    -1.0 * airborne,
        }
        if self._standing():
            # stand still: all four feet planted and joints quiet (no trotting in place)
            terms["stand_feet"] = 1.0 * float(contact.all())
            # bounded (max -1): a stand command can arrive mid-stride with fast joints, and an
            # unbounded penalty made falling over the cheapest way out
            terms["stand_still"] = -min(0.005 * np.sum(dq ** 2), 1.0)
        return {k: float(v) for k, v in terms.items()}

    def close(self):
        """Shut down PyBullet."""
        self.pb.disconnect()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    for omni in (False, True):
        env = Lite3Env(omni=omni)
        try:
            check_env(env, skip_render_check=True)
        except AssertionError as e:   # PyBullet reset isn't bit-exact deterministic; not fatal
            print("check_env warning:", e)

        # self-check 1: zero action = hold default pose -> must stay standing for 5 s
        obs, _ = env.reset(seed=0)
        for _ in range(250):
            obs, r, term, trunc, info = env.step(np.zeros(12))
            assert not term, "robot fell while holding the default pose"
        h = env.pb.getBasePositionAndOrientation(env.robot)[0][2]
        print(f"omni={omni}: standing height after 5 s: {h:.3f} m  (BASE_HEIGHT = {BASE_HEIGHT})")
        print("body contacts while standing:", info["body_contact"], "(should be 0)")

        # self-check 2: random actions give finite obs/reward and episodes end
        obs, _ = env.reset(seed=1)
        for _ in range(2000):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            if term or trunc:
                obs, _ = env.reset()
        env.close()
    print("ok")
