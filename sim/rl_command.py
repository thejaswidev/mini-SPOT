"""Train / evaluate / drive the ONE policy for all movement commands (paper architecture).

    python -m sim.rl_command train                     # PPO, 8 parallel sims, saves models/command_ppo.zip
    python -m sim.rl_command train --steps 1000000     # shorter run
    python -m sim.rl_command eval                      # plain paper gait vs gait + policy, per command
    python -m sim.rl_command eval --scenario mixed     # same on bumps + slopes
    python -m sim.rl_command drive                     # 3D viewer, type commands (macOS: use mjpython)
    python -m sim.rl_command drive --plain             # same with the plain paper gait (no network needed)
    python -m sim.rl_command drive --scenario bumps6   # drive on bumpy ground

Drive commands (type + Enter):  w forward   s backward   a strafe left   d strafe right
                                q turn left   e turn right   x stop   (or the full names)

Ctrl+C during training saves to models/command_ppo_interrupted; checkpoints every 250 k steps
(models/command_ckpt_*) can be loaded with --model.
See sim/command_env.py (environment) and sim/bezier3d.py (3D Bezier gait + IK).
"""
import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np

from sim.command_env import COMMANDS, CommandGaitEnv
from sim.rl_residual import SCENARIOS, load_policy, train_ppo

MODEL_PATH = Path("models") / "command_ppo"

# typical command used for eval / drive: the middle of each training range
TYPICAL = {name: tuple(float(np.mean(r)) for r in rng) for name, rng in COMMANDS.items()}
KEYS = {"w": "forward", "s": "backward", "a": "strafe_left", "d": "strafe_right",
        "q": "turn_left", "e": "turn_right", "x": "stand", "stop": "stand",
        "left": "strafe_left", "right": "strafe_right",
        "turn left": "turn_left", "turn right": "turn_right"}


def train(total_steps, n_envs):
    train_ppo(CommandGaitEnv, total_steps, n_envs, MODEL_PATH)


def plain_policy():
    return lambda obs: np.zeros(14)                  # action 0 = plain paper gait


def run(act, command, scenario, episodes):
    """Returns (episodes survived, mean achieved (vx, vy, wz) after the first second)."""
    alive, vels = 0, []
    for ep in range(episodes):
        env = CommandGaitEnv(command=TYPICAL[command], **SCENARIOS[scenario])
        obs, _ = env.reset(seed=2000 + ep)
        t0 = env.data.time
        while True:
            obs, _, term, trunc, info = env.step(act(obs))
            if env.data.time - t0 > 1.0:
                vels.append(info["vel"])
            if term or trunc:
                break
        alive += not term
        env.close()
    return alive, np.mean(vels, axis=0)


def evaluate(scenario, episodes, model_path, commands):
    act = load_policy(model_path)
    print(f"scenario '{scenario}', {episodes} episodes of 10 s per command; "
          f"achieved (vx m/s, vy m/s, wz rad/s)")
    print(f"{'command':>13} {'asked':>20} | {'plain gait':>30} | {'gait + policy':>30}")
    fmt = lambda v: f"({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})"
    for name in commands:
        base = run(plain_policy(), name, scenario, episodes)
        pol = run(act, name, scenario, episodes)
        print(f"{name:>13} {fmt(TYPICAL[name]):>20} | {base[0]:>2}/{episodes} up {fmt(base[1]):>20} | "
              f"{pol[0]:>2}/{episodes} up {fmt(pol[1]):>20}")


def drive(scenario, model_path, plain):
    env = CommandGaitEnv(render_mode="human", command=TYPICAL["stand"],
                         **dict(SCENARIOS[scenario], max_episode_s=1e9))
    act = plain_policy() if plain else load_policy(model_path)
    obs, _ = env.reset()
    print(__doc__.split("Drive commands")[1].split("Ctrl+C")[0].strip())
    print(">> standing - type a command")
    while True:
        t0 = time.time()
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower().replace("_", " ")
            name = KEYS.get(line, line.replace(" ", "_"))
            if name in TYPICAL:
                env.set_command(TYPICAL[name], name)
                print(f">> {name}  (vx, vy, wz) = {TYPICAL[name]}")
            elif line:
                print(">> unknown command; use w a s d q e x")
        obs, _, term, trunc, info = env.step(act(obs))
        if term:
            print(">> fell - resetting")
            obs, _ = env.reset()
            env.set_command(TYPICAL["stand"], "stand")
        time.sleep(max(0.0, env.control_dt - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval", "drive"))
    parser.add_argument("--steps", type=int, default=5_000_000, help="train: total control steps")
    parser.add_argument("--envs", type=int, default=8, help="train: parallel simulations")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="flat",
                        help="eval/drive: terrain (training always uses the full random mix)")
    parser.add_argument("--commands", nargs="+", choices=list(COMMANDS), default=list(COMMANDS),
                        help="eval: which commands")
    parser.add_argument("--episodes", type=int, default=6, help="eval: episodes per command")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="eval/drive: model or checkpoint")
    parser.add_argument("--plain", action="store_true", help="drive: plain paper gait, no network")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.steps, args.envs)
    elif args.mode == "eval":
        evaluate(args.scenario, args.episodes, args.model, args.commands)
    else:
        drive(args.scenario, args.model, args.plain)


if __name__ == "__main__":
    main()
