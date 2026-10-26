"""Train / evaluate the RL curve modulation (Bezier gait reshaped slightly by a policy).

    python -m sim.rl_residual train                         # PPO, 8 parallel sims, saves models/residual_ppo.zip
    python -m sim.rl_residual train --steps 1000000         # shorter run
    python -m sim.rl_residual eval                          # Bezier only vs Bezier + policy, per scenario
    python -m sim.rl_residual eval --scenarios uphill15 bumps6 --episodes 20
    python -m sim.rl_residual watch --scenario mixed        # viewer (on macOS run with mjpython)
    python -m sim.rl_residual watch --scenario bumps6 --plain   # same, plain Bezier gait (no network)
    python -m sim.rl_residual eval --model models/residual_ckpt_2000000_steps   # use a checkpoint
    python -m sim.rl_residual train --gait-params models/best_params.json   # build on rl_ppo's best gait

Ctrl+C during training saves to models/residual_ppo_interrupted, so a finished model is never
overwritten by a partial one. Every checkpoint saves its own normalisation file next to it.

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

_OFF = dict(bump_height=(0.0, 0.0), slope_deg=(0.0, 0.0), side_slope_deg=(0.0, 0.0), push_force=(0.0, 0.0))
SCENARIOS = {                     # fixed test conditions (robot dynamics still randomized)
    "flat":      dict(_OFF),
    "bumps4":    dict(_OFF, bump_height=(0.04, 0.04)),
    "bumps6":    dict(_OFF, bump_height=(0.06, 0.06)),
    "uphill10":  dict(_OFF, slope_deg=(10.0, 10.0)),
    "uphill15":  dict(_OFF, slope_deg=(15.0, 15.0)),
    "downhill15": dict(_OFF, slope_deg=(-15.0, -15.0)),
    "side8":     dict(_OFF, side_slope_deg=(8.0, 8.0)),
    "push80":    dict(_OFF, push_force=(80.0, 80.0)),
    "mixed":     dict(),          # the training distribution: everything at once
}


def train(total_steps, n_envs, gait_params):
    train_ppo(partial(ResidualGaitEnv, gait_params=gait_params), total_steps, n_envs, MODEL_PATH)


def train_ppo(env_factory, total_steps, n_envs, model_path):
    """PPO with parallel sims. Saves <model_path>.zip + its normalisation file, a checkpoint every
    250 k steps (<name>_ckpt_*), and on Ctrl+C <model_path>_interrupted instead of model_path."""
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = model_path.name.removesuffix("_ppo") + "_ckpt"
    venv = make_vec_env(env_factory, n_envs=n_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=1024, batch_size=min(4096, 1024 * n_envs), n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),   # start with small corrections
    )
    checkpoint = CheckpointCallback(save_freq=max(1, 250_000 // n_envs), save_path=str(model_path.parent),
                                    name_prefix=prefix, save_vecnormalize=True)
    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint)
    except KeyboardInterrupt:
        model_path = model_path.with_name(model_path.name + "_interrupted")
        print(f"\ninterrupted at {model.num_timesteps} steps")
    finally:
        model.save(str(model_path))
        venv.save(str(norm_path_for(model_path)))
        venv.close()
        print(f"saved {model_path}.zip and {norm_path_for(model_path)}")


def norm_path_for(model_path):
    """The observation-normalisation file that belongs to a model / checkpoint."""
    model_path = Path(str(model_path).removesuffix(".zip"))
    if model_path == MODEL_PATH:
        return NORM_PATH
    if "_ckpt_" in model_path.name:                      # SB3 checkpoint naming
        prefix, steps = model_path.name.split("_ckpt_", 1)
        return model_path.with_name(f"{prefix}_ckpt_vecnormalize_{steps}.pkl")
    return model_path.with_name(model_path.name + "_vecnormalize.pkl")


def load_policy(model_path=MODEL_PATH):
    """Returns act(obs) -> action, using the saved obs normalisation."""
    import pickle
    from stable_baselines3 import PPO

    model_path = Path(str(model_path).removesuffix(".zip"))
    norm_path = norm_path_for(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} is missing: it holds the input normalisation for "
                                f"{model_path}.zip (checkpoints from before this fix don't have one)")
    model = PPO.load(str(model_path), device="cpu")
    with open(norm_path, "rb") as f:
        norm = pickle.load(f)
    rms, clip, eps = norm.obs_rms, norm.clip_obs, norm.epsilon

    def act(obs):
        o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
        return model.predict(o.astype(np.float32), deterministic=True)[0]
    return act


def run_episodes(act, scenario, episodes, gait_params):
    falls, dist = 0, []
    for ep in range(episodes):
        env = ResidualGaitEnv(gait_params=gait_params, **SCENARIOS[scenario])
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


def evaluate(scenarios, episodes, gait_params, model_path):
    act = load_policy(model_path)
    env = ResidualGaitEnv(gait_params=gait_params)
    act_dim = env.act_dim
    env.close()
    plain = lambda obs: np.zeros(act_dim)          # action 0 = plain Bezier curve
    print(f"{episodes} episodes of 10 s per row, randomized robot; distance = forward progress")
    print(f"{'scenario':>11} | {'Bezier only':>22} | {'Bezier + policy':>22}")
    for name in scenarios:
        base = run_episodes(plain, name, episodes, gait_params)
        pol = run_episodes(act, name, episodes, gait_params)
        print(f"{name:>11} | survived {base[0]:>2}/{episodes}, {base[1]:+.2f} m | "
              f"survived {pol[0]:>2}/{episodes}, {pol[1]:+.2f} m")


def watch(scenario, gait_params, model_path, plain=False):
    import time
    env = ResidualGaitEnv(gait_params=gait_params, render_mode="human", **SCENARIOS[scenario])
    act = (lambda obs: np.zeros(env.act_dim)) if plain else load_policy(model_path)
    obs, _ = env.reset()
    while True:
        t0 = time.time()
        obs, _, term, trunc, info = env.step(act(obs))
        if term or trunc:
            print("fell" if term else "episode done", "- pushes:", info["pushes"],
                  "- slope (deg, up/side):", tuple(round(a, 1) for a in info["slope"]))
            obs, _ = env.reset()
        time.sleep(max(0.0, env.control_dt - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval", "watch"))
    parser.add_argument("--steps", type=int, default=3_000_000, help="train: total control steps")
    parser.add_argument("--envs", type=int, default=8, help="train: parallel simulations")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                        help="eval: test conditions")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="mixed", help="watch: condition")
    parser.add_argument("--plain", action="store_true", help="watch: plain Bezier gait, no network")
    parser.add_argument("--episodes", type=int, default=12, help="eval: episodes per scenario")
    parser.add_argument("--gait-params", type=str, default=None,
                        help="best_params.json from sim.rl_ppo (default: hand-tuned gait)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="eval/watch: model or checkpoint to load (default models/residual_ppo)")
    args = parser.parse_args()
    gait_params = load_gait_params(args.gait_params) if args.gait_params else None

    if args.mode == "train":
        train(args.steps, args.envs, gait_params)
    elif args.mode == "eval":
        evaluate(args.scenarios, args.episodes, gait_params, args.model)
    else:
        watch(args.scenario, gait_params, args.model, args.plain)


if __name__ == "__main__":
    main()
