"""Train (fine-tune) the navigation policy with PPO.

The navigation policy looks through the robot's camera and sends velocity commands to the
frozen walking policy (models/walking_policy.zip). Training continues from an existing
navigation policy: 30% of episodes on the predefined Q2 map (random starts), 70% on random maps.
example: python train_navigation.py 8000000 8 --from models/navigation_policy.zip
Output: models/navigation_policy.zip 
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE) 
import numpy as np
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from q2_navigation_env import Q2NavigationEnv

WALKER = os.path.join(HERE, "models/walking_policy.zip")   # frozen, only the navigation policy learns


# how the parallel simulators are started: "fork" is fast but Linux-only
START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"   # Windows: spawn

class SuccessRate(BaseCallback):
    """Logs the share of the last 200 finished episodes that reached the goal (and the obstacle
    count, fixed at 20 here)."""
    def __init__(self, window=200, max_obstacles=20):
        super().__init__()
        self.results = deque(maxlen=window)
        self.max_obstacles = max_obstacles

    def _on_step(self):
        # an episode just ended in some simulator: remember whether it reached the goal
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.results.append(float(info.get("success", False)))
        if len(self.results) == self.results.maxlen:
            self.logger.record("nav/success_rate", float(np.mean(self.results)))
            self.logger.record("nav/max_obstacles", self.max_obstacles)
        return True


class MakeEnv:
    def __call__(self):
        # Q2 settings: 4 m goal, colour image to 20 m, depth 0.28-20 m, 30% of episodes on the map
        return Q2NavigationEnv(WALKER, max_obstacles=20, target_height=4.0, render_far=20.0,
                          depth_range=(0.28, 20.0), fixed_map_frac=0.3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=6_000_000, help="amount of environment steps to train for")
    ap.add_argument("n_envs", nargs="?", type=int, default=48, help="parallel simulators (about the number of CPU cores)")
    ap.add_argument("--from", dest="src", default=os.path.join(HERE, "models/navigation_policy_before_q2.zip"),
                    help="navigation policy to start from")
    args = ap.parse_args()
    os.chdir(HERE)
    if not os.path.exists(args.src):
        sys.exit(f"starting model {args.src} not found. The Q2 fine-tune starts from models/navigation_policy_before_q2.zip "
                 f"(the navigation policy before Q2); to keep training the final policy instead use "
                 f"--from models/navigation_policy.zip")
    # n_envs simulators in separate processes; VecMonitor logs episode reward and length
    env = VecMonitor(SubprocVecEnv([MakeEnv() for _ in range(args.n_envs)], start_method=START_METHOD))
    model = PPO.load(args.src, env=env, device="cpu") # same network, keeps learning from here
    with torch.no_grad():  # a little fresh exploration for the new setting
        model.policy.log_std.clamp_(min=float(np.log(0.25)))
    # learn, save a checkpoint every 500k steps
    model.learn(args.steps, reset_num_timesteps=False, callback=CallbackList([
        SuccessRate(), CheckpointCallback(max(1, 500_000 // args.n_envs), "checkpoints", "navigation_policy")]))
    os.makedirs("models", exist_ok=True)
    model.save("models/navigation_policy")
    print("saved models/navigation_policy.zip", flush=True)
