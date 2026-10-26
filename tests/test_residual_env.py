"""Checks for sim/residual_env.py: IMU-only observations, small curve changes, terrain + slopes."""

import numpy as np
import pytest

from sim.bezier_helpers import compute_leg
from sim.residual_env import DEFAULT_GAIT_PARAMS, ResidualGaitEnv, modulated_leg

FLAT = dict(bump_height=(0.0, 0.0), slope_deg=(0.0, 0.0), side_slope_deg=(0.0, 0.0),
            push_force=(0.0, 0.0))


def run(env, seed=0):
    obs, _ = env.reset(seed=seed)
    x0 = env.data.qpos[0]
    for _ in range(env.max_steps):
        obs, reward, term, trunc, info = env.step(np.zeros(env.act_dim))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if term or trunc:
            break
    return term, env.data.qpos[0] - x0, info


@pytest.mark.parametrize("phase", np.linspace(0, 0.99, 12))
def test_zero_modulation_is_the_plain_bezier_curve(phase):
    p = DEFAULT_GAIT_PARAMS
    assert np.allclose(modulated_leg(p, 0.35, phase, 0.45), compute_leg(p, 0.35, phase, 0.45))


def test_observation_is_imu_only():
    env = ResidualGaitEnv()
    obs, _ = env.reset(seed=0)
    # gravity 3 + gyro 3 + phase 2 + previous action
    assert obs.shape == (8 + env.act_dim,)
    assert env.act_dim == 5 * 4            # 5 curve changes per leg (with HAA)


def test_plain_gait_walks_on_flat_ground():
    fell, dx, _ = run(ResidualGaitEnv(randomize=False, **FLAT))
    assert not fell and dx > 0.8           # the plain trot covers ~1.5 m in 10 s


def test_terrain_has_bumps_and_start_is_at_ground_level():
    env = ResidualGaitEnv(randomize=False, **dict(FLAT, bump_height=(0.04, 0.04)))
    env.reset(seed=0)
    heights = [env._ground_height(x, y) for x in np.linspace(-1, 1, 21) for y in np.linspace(-1, 1, 21)]
    assert np.ptp(heights) > 0.02
    assert abs(env._ground_height(0.0, 0.0)) < 1e-9


def test_uphill_tilts_gravity_backwards():
    env = ResidualGaitEnv(randomize=False, **dict(FLAT, slope_deg=(10.0, 10.0)))
    env.reset(seed=0)
    g = env.model.opt.gravity
    assert g[0] < 0 and np.isclose(np.linalg.norm(g), 9.81)


def test_pushes_are_applied():
    _, _, info = run(ResidualGaitEnv(randomize=False, **dict(FLAT, push_force=(40.0, 40.0))))
    assert info["pushes"] >= 2
