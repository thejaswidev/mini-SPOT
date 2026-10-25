"""Train / evaluate the closed-loop residual policy (Bezier gait + learned corrections).

    python -m sim.rl_residual train                         # PPO, 8 parallel sims, saves models/residual_ppo.zip
    python -m sim.rl_residual train --steps 3000000
    python -m sim.rl_residual eval                          # Bezier only vs Bezier + policy under pushes
    python -m sim.rl_residual eval --push 100 --episodes 20
    python -m sim.rl_residual watch                         # viewer (on macOS run with mjpython)
    python -m sim.rl_residual train --gait-params models/best_params.json   # build on rl_ppo's best gait

One step = one control tick (50 Hz). See sim/residual_env.py for obs / action / reward.
"""
import argparse
from functools import partial
from pathlib import Path

import numpy as np

from sim.residual_env import ResidualGaitEnv, load_gait_params

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "residual_ppo"
NORM_PATH = MODELS_DIR / "residual_vecnormalize.pkl"


def train(total_steps, n_envs, gait_params):
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(partial(ResidualGaitEnv, gait_params=gait_params), n_envs=n_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=1024, batch_size=min(4096, 1024 * n_envs), n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),   # start with small corrections
    )
    checkpoint = CheckpointCallback(save_freq=max(1, 250_000 // n_envs), save_path=str(MODELS_DIR),
                                    name_prefix="residual_ckpt")
    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint)
    finally:
        model.save(str(MODEL_PATH))
        venv.save(str(NORM_PATH))
        venv.close()
        print(f"saved {MODEL_PATH}.zip and {NORM_PATH}")


def load_policy():
    """Returns act(obs) -> action, using the saved obs normalisation."""
    import pickle
    from stable_baselines3 import PPO

    model = PPO.load(str(MODEL_PATH), device="cpu")
    with open(NORM_PATH, "rb") as f:
        norm = pickle.load(f)
    rms, clip, eps = norm.obs_rms, norm.clip_obs, norm.epsilon

    def act(obs):
        o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
        return model.predict(o.astype(np.float32), deterministic=True)[0]
    return act


def run_episodes(act, push, episodes, gait_params, randomize=True):
    falls, dist = 0, []
    for ep in range(episodes):
        env = ResidualGaitEnv(gait_params=gait_params, randomize=randomize, push_force=(push, push))
        obs, _ = env.reset(seed=1000 + ep)
        x0 = env.data.qpos[0]
        while True:
            obs, _, term, trunc, _ = env.step(act(obs))
            if term or trunc:
                break
        falls += term
        dist.append(env.data.qpos[0] - x0)
        env.close()
    return episodes - falls, float(np.mean(dist))


def evaluate(pushes, episodes, gait_params):
    act = load_policy()
    print(f"{episodes} episodes of 10 s per row, one push every 1.5-3 s, randomized robot")
    print(f"{'push':>6} | {'Bezier only':>22} | {'Bezier + policy':>22}")
    for push in pushes:
        env = ResidualGaitEnv(gait_params=gait_params)
        nu = env.nu
        env.close()
        base = run_episodes(lambda o: np.zeros(nu), push, episodes, gait_params)
        pol = run_episodes(act, push, episodes, gait_params)
        print(f"{push:>4} N | survived {base[0]:>2}/{episodes}, {base[1]:+.2f} m | "
              f"survived {pol[0]:>2}/{episodes}, {pol[1]:+.2f} m")


def watch(push, gait_params):
    import time
    act = load_policy()
    env = ResidualGaitEnv(gait_params=gait_params, render_mode="human", push_force=(push, push))
    obs, _ = env.reset()
    while True:
        t0 = time.time()
        obs, _, term, trunc, info = env.step(act(obs))
        if term or trunc:
            print("fell" if term else "episode done", "- pushes:", info["pushes"])
            obs, _ = env.reset()
        time.sleep(max(0.0, env.control_dt - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval", "watch"))
    parser.add_argument("--steps", type=int, default=2_000_000, help="train: total control steps")
    parser.add_argument("--envs", type=int, default=8, help="train: parallel simulations")
    parser.add_argument("--push", type=float, nargs="+", default=[60, 80, 100],
                        help="eval/watch: push force(s) in N")
    parser.add_argument("--episodes", type=int, default=12, help="eval: episodes per push force")
    parser.add_argument("--gait-params", type=str, default=None,
                        help="best_params.json from sim.rl_ppo (default: hand-tuned gait)")
    args = parser.parse_args()
    gait_params = load_gait_params(args.gait_params) if args.gait_params else None

    if args.mode == "train":
        train(args.steps, args.envs, gait_params)
    elif args.mode == "eval":
        evaluate(args.push, args.episodes, gait_params)
    else:
        watch(args.push[0], gait_params)


if __name__ == "__main__":
    main()
