"""Test or train the PPO policy for the Mini-SPOT Bezier gait.

One policy action = one full episode: the gait parameters are set once, the sim
runs with them, and the summed reward comes back.

Training logs every episode of every env to models/episodes.csv, so the global
best parameters and the learning curve can be recovered afterwards
(models/best_params.json is written at the end of training).
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from sim.gym_class import QuadrupedBezierEnv
from sim.bezier_helpers import PARAM_NAMES

MODELS_DIR = Path("models")
LOG_CSV = MODELS_DIR / "episodes.csv"
BEST_JSON = MODELS_DIR / "best_params.json"
N_ENVS = 6


def fmt_params(params):
    parts = []
    for name, val in zip(PARAM_NAMES, params):
        unit = "s" if "period" in name else "rad"
        parts.append(f"{name}={val:+.3f}{unit}")
    return "  ".join(parts)


class EpisodeWrapper(gym.Wrapper):
    """One step == one episode.

    - Forces terminated/truncated so the vec env always resets after one step.
    - Optionally prints parameters before the sim and the result after.
    - Optionally appends every episode to a shared CSV (safe across processes,
      each row is a single small append).
    """

    def __init__(self, env, verbose=True, window=10, log_path=None):
        super().__init__(env)
        self.verbose = verbose
        self.window = window
        self.log_path = log_path
        self.episode = 0
        self.returns = []
        self.best = -np.inf
        self.best_params = None

    def step(self, action):
        self.episode += 1
        base = self.env.unwrapped
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        params = base._to_phys(a)

        if self.verbose:
            print(f"\n=== Episode {self.episode} (pid {os.getpid()}) ===")
            print(f"  action (norm): {np.array2string(a, precision=3, suppress_small=True)}")
            print(f"  params set   : {fmt_params(params)}")
            print(f"  running sim for up to {base.max_steps * base.n_control * base.control_dt:.1f}s ...")

        t0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        wall = time.time() - t0

        # single-step episodes: always signal the end of the episode
        if not terminated:
            truncated = True

        self.returns.append(reward)
        if reward > self.best:
            self.best = reward
            self.best_params = params.copy()

        if self.log_path is not None:
            row = [os.getpid(), self.episode, f"{reward:.6f}", int(terminated),
                   f"{info['forward_vel']:.4f}", f"{info['distance']:.4f}"]
            row += [f"{p:.6f}" for p in params]
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        if self.verbose:
            recent = float(np.mean(self.returns[-self.window:]))
            outcome = "FELL / terminated early" if terminated else "survived full episode"
            print(f"  result       : {outcome}  (wall time {wall:.1f}s)")
            print(f"  reward       : {reward:+.3f}   (mean last {self.window}: {recent:+.3f})")
            print(f"  fwd vel      : {info['forward_vel']:+.3f} m/s   distance: {info['distance']:.2f} m")
            terms = "  ".join(f"{k}={v:+.2f}" for k, v in info["reward_terms"].items())
            print(f"  reward terms : {terms}")
            print(f"  best so far  : {self.best:+.3f}  with  {fmt_params(self.best_params)}")
        return obs, reward, terminated, truncated, info


def init_log():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["pid", "episode", "reward", "terminated", "fwd_vel", "distance"] + list(PARAM_NAMES)
        )


def save_best_from_log():
    """Find the globally best episode in the CSV and write best_params.json."""
    if not LOG_CSV.exists():
        return
    with open(LOG_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["reward"]))
    out = {
        "reward": float(best["reward"]),
        "fwd_vel": float(best["fwd_vel"]),
        "distance": float(best["distance"]),
        "params": {name: float(best[name]) for name in PARAM_NAMES},
    }
    BEST_JSON.write_text(json.dumps(out, indent=2))
    print(f"best reward {out['reward']:+.3f} -> saved {BEST_JSON}")
    print("  " + fmt_params([out["params"][n] for n in PARAM_NAMES]))


def run_test(n_episodes, render, model_path=None):
    env = QuadrupedBezierEnv(render_mode="human" if render else None)
    env = EpisodeWrapper(env, verbose=True)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape)

    model = None
    if model_path:
        from stable_baselines3 import PPO

        model = PPO.load(model_path)  # accepts the path with or without ".zip"
        print(f"loaded model: {model_path}")
    else:
        print("no --model given: using a zero action (midpoint of the param ranges)")

    for _ in range(n_episodes):
        obs, _ = env.reset()
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(action)

    if env.best_params is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        BEST_JSON.write_text(json.dumps(
            {"reward": float(env.best),
             "params": {n: float(v) for n, v in zip(PARAM_NAMES, env.best_params)}},
            indent=2))
        print(f"saved best test params to {BEST_JSON}")
    env.close()


def train(total_episodes, n_steps, verbose):
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    class SaveRolloutCheckpointCallback(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            checkpoint_path = MODELS_DIR / f"ppo_{self.num_timesteps}_steps"
            self.model.save(str(checkpoint_path))
            print(f"saved checkpoint: {checkpoint_path}.zip")
            return True

    init_log()
    log_path = str(LOG_CSV)

    def make_env():
        # render_mode=None: no viewer during training
        return EpisodeWrapper(QuadrupedBezierEnv(render_mode=None),
                              verbose=verbose, log_path=log_path)

    # SubprocVecEnv: the 6 envs really run in parallel (DummyVecEnv would be sequential)
    venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    # normalise rewards only (observation is not informative for a one-step problem)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.0)
    print("obs shape:", venv.observation_space.shape)

    buffer_size = n_steps * N_ENVS  # episodes per PPO update
    batch_size = min(128, buffer_size)
    while buffer_size % batch_size != 0:
        batch_size -= 1
    print(f"episodes per update: {buffer_size}, minibatch: {batch_size}, "
          f"updates: ~{int(np.ceil(total_episodes / buffer_size))}")

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=n_steps,            # steps (=episodes) per env per update
        batch_size=batch_size,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.0,                  # one-step episodes: no bootstrapping
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs=dict(log_std_init=-1.0),   # initial action std ~0.37
    )
    print("env initialised")

    # one timestep == one episode, summed over all envs (SB3 counts total timesteps).
    # SB3 rounds up to a whole number of rollouts.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model.learn(total_timesteps=total_episodes, callback=SaveRolloutCheckpointCallback())
    finally:
        model.save("quadruped_bezier_ppo")
        venv.save(str(MODELS_DIR / "vecnormalize.pkl"))
        print("saved quadruped_bezier_ppo and models/vecnormalize.pkl")
        save_best_from_log()
        venv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "train"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="train: total episodes over all envs (default 20000); "
                             "test: episodes to run (default 3)")
    parser.add_argument("--n-steps", type=int, default=128,
                        help="train: episodes per env per PPO update (default 128; "
                             "one update uses 128 x 6 = 768 episodes)")
    parser.add_argument("--no-render", action="store_true", help="test: disable the viewer")
    parser.add_argument("--verbose", action="store_true",
                        help="train: print every episode (noisy with 6 parallel envs)")
    parser.add_argument("--model", type=str, default=None,
                        help="test: path to a trained model, e.g. quadruped_bezier_ppo (.zip optional)")
    args = parser.parse_args()

    if args.mode == "test":
        run_test(args.episodes or 3, render=not args.no_render, model_path=args.model)
    else:
        train(args.episodes or 20000, args.n_steps, args.verbose)


if __name__ == "__main__":
    main()