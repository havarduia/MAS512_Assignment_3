"""Train the Lite3 walking policy with PPO.

The walking policy turns a velocity command (forward, sideways, turn) into 12 joint targets.
It is trained on its own first; the navigation policy then drives it (see train_navigation.py).

example: python train_walking.py 2000000 8 --omni
Output: walking_policy_new.zip 
"""
import argparse
import sys
import numpy as np
import torch
torch.set_num_threads(1)  # default is 64 threads for a tiny MLP
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from walking_env import Lite3Env


# how the parallel simulators are started: "fork" is fast but Linux-only
START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"   # Windows spawn

class MakeEnv:
    """Picklable env factory for SubprocVecEnv."""
    def __init__(self, omni, rough): #omni means omni-directional, rough means rough terrain"
        self.omni, self.rough = omni, rough

    def __call__(self):
        # every parallel simulator gets its own walking environment (own PyBullet instance)
        return Lite3Env(omni=self.omni, rough=self.rough)


class Difficulty(BaseCallback):
    """Rough mode: difficulty 0.2 to 1.0 over the first half of this run, then held at 1."""
    def __init__(self, total):
        super().__init__()
        self.total, self.start, self.level = total, None, -1.0

    def _on_step(self):
        # called by PPO after every step: raise the terrain difficulty in all simulators
        if self.start is None:
            self.start = self.num_timesteps
        level = round(min(1.0, 0.2 + 1.6 * (self.num_timesteps - self.start) / self.total), 2)
        if level != self.level:
            self.level = level
            self.training_env.set_attr("difficulty", level)
            self.logger.record("rough/difficulty", level)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=300_000, help="amount of environment steps to train for")
    ap.add_argument("n_envs", nargs="?", type=int, default=8, help="parallel simulators (about the number of CPU cores)")
    ap.add_argument("resume", nargs="?", default=None, help="continue training this policy .zip instead of starting fresh")
    ap.add_argument("--omni", action="store_true", help="train all directions + standing, not only forward")
    ap.add_argument("--rough", action="store_true", help="rough ground, steps, pushes, friction and payload changes")
    ap.add_argument("--reset-std", type=float, default=None, metavar="S",
                    help="when resuming: set the exploration noise back to S (e.g. 0.15)")
    args = ap.parse_args()
    print(f"using {args.n_envs} envs, omni={args.omni}, rough={args.rough}", flush=True)

    # n_envs simulators in separate processes; VecMonitor logs episode reward and length
    env = VecMonitor(SubprocVecEnv([MakeEnv(args.omni, args.rough) for _ in range(args.n_envs)], start_method=START_METHOD))
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        print("resumed from", args.resume, "at", model.num_timesteps, "steps", flush=True)
        if args.reset_std:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(args.reset_std)))
            print(f"exploration noise reset to std {args.reset_std}", flush=True)
    else:
        # new policy: MLP 256-128-64; 512 steps per simulator are collected before each update
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=2048, learning_rate=3e-4,
                    gamma=0.99, policy_kwargs=dict(net_arch=[256, 128, 64], log_std_init=-1.0),
                    verbose=1, device="cpu")

    # save a checkpoint every 200k steps, so a crash or a bad end doesn't lose the run
    callbacks = [CheckpointCallback(max(1, 200_000 // args.n_envs), "checkpoints_walking", "walking")]
    if args.rough:
        callbacks.append(Difficulty(args.steps))
    model.learn(args.steps, callback=CallbackList(callbacks), reset_num_timesteps=args.resume is None)
    model.save("walking_policy_new")
    print("saved walking_policy_new.zip", flush=True)
