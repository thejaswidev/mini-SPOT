"""Checks for sim/residual_env.py: sensor-only observations, zero action = plain Bezier gait."""

import numpy as np

from sim.residual_env import ResidualGaitEnv


def run(env, steps=None):
    obs, _ = env.reset(seed=0)
    x0 = env.data.qpos[0]
    for _ in range(steps or env.max_steps):
        obs, reward, term, trunc, info = env.step(np.zeros(env.nu))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if term or trunc:
            break
    return term, env.data.qpos[0] - x0, info


def test_observation_is_sensor_only():
    env = ResidualGaitEnv()
    obs, _ = env.reset(seed=0)
    # gravity 3 + gyro 3 + foot contacts 4 + phase 2 + previous action (one per actuator)
    assert obs.shape == (12 + env.nu,)
    assert env.action_space.shape == (env.nu,)


def test_zero_action_walks_forward_without_pushes():
    env = ResidualGaitEnv(randomize=False, push_force=(0.0, 0.0))
    fell, dx, _ = run(env)
    assert not fell
    assert dx > 0.8                      # the plain trot covers ~1.5 m in 10 s


def test_pushes_are_applied():
    env = ResidualGaitEnv(randomize=False, push_force=(40.0, 40.0))
    _, _, info = run(env)
    assert info["pushes"] >= 2
