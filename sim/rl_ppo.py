"""Test or train the PPO policy for the Mini-SPOT Bezier gait.

One policy action = one full episode: the gait parameters are set once, the sim
runs with them, and the summed reward comes back. Every episode prints the
parameters that were set (before the sim runs) and the result (after).
"""
## does not yet save best parameters
import argparse
import time

import numpy as np
import gymnasium as gym

from sim.gym_class import QuadrupedBezierEnv
from sim.bezier_helpers import PARAM_NAMES


def fmt_params(params):
    parts = []
    for name, val in zip(PARAM_NAMES, params):
        unit = "s" if "period" in name else "rad"
        parts.append(f"{name}={val:+.3f}{unit}")
    return "  ".join(parts)


class VerboseEpisodeWrapper(gym.Wrapper):
    """Prints the parameters before the sim runs, then the result of the episode."""

    def __init__(self, env, window=10):
        super().__init__(env)
        self.episode = 0
        self.window = window
        self.returns = []
        self.best = -np.inf
        self.best_params = None

    def step(self, action):
        self.episode += 1
        base = self.env.unwrapped
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        params = base._to_phys(a)

        print(f"\n=== Episode {self.episode} ===")
        print(f"  action (norm): {np.array2string(a, precision=3, suppress_small=True)}")
        print(f"  params set   : {fmt_params(params)}")
        print(f"  running sim for up to {base.max_steps * base.n_control * base.control_dt:.1f}s ...")

        t0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        wall = time.time() - t0

        self.returns.append(reward)
        recent = float(np.mean(self.returns[-self.window:]))
        if reward > self.best:
            self.best = reward
            self.best_params = params.copy()

        outcome = "FELL / terminated early" if terminated else "survived full episode"
        print(f"  result       : {outcome}  (wall time {wall:.1f}s)")
        print(f"  reward       : {reward:+.3f}   (mean last {self.window}: {recent:+.3f})")
        print(f"  fwd vel      : {info['forward_vel']:+.3f} m/s   distance: {info['distance']:.2f} m")
        terms = "  ".join(f"{k}={v:+.2f}" for k, v in info["reward_terms"].items())
        print(f"  reward terms : {terms}")
        print(f"  best so far  : {self.best:+.3f}  with  {fmt_params(self.best_params)}")
        return obs, reward, terminated, truncated, info


def run_test(n_episodes, render, model_path=None):
    env = QuadrupedBezierEnv(render_mode="human" if render else None)
    env = VerboseEpisodeWrapper(env)
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
            action = np.zeros(6, dtype=np.float32)
        env.step(action)
    env.close()


def train(total_episodes, n_steps):
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    print("import done")
    # render_mode=None: no viewer during training (it would open a window and slow things down)
    venv = make_vec_env(
        lambda: VerboseEpisodeWrapper(QuadrupedBezierEnv(render_mode=None)),
        n_envs=1,
    )
    print("obs shape:", venv.observation_space.shape)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=n_steps,        # episodes collected per PPO update (1 step = 1 episode)
        batch_size=n_steps,
        learning_rate=3e-4,
        gamma=0.97,
        ent_coef=0.005,
    )
    print("env initialised")
    # one timestep == one episode, so this is the number of episodes
    model.learn(total_timesteps=total_episodes)
    model.save("quadruped_bezier_ppo")
    print("saved quadruped_bezier_ppo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "train"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="train: total episodes (default 2000); test: episodes to run (default 3)")
    parser.add_argument("--n-steps", type=int, default=64,
                        help="train: episodes per PPO update (default 64)")
    parser.add_argument("--no-render", action="store_true", help="test: disable the viewer")
    parser.add_argument("--model", type=str, default=None,
                        help="test: path to a trained model, e.g. quadruped_bezier_ppo (.zip optional)")
    args = parser.parse_args()

    if args.mode == "test":
        run_test(args.episodes or 3, render=not args.no_render, model_path=args.model)
    else:
        train(args.episodes or 2000, args.n_steps)


if __name__ == "__main__":
    main()