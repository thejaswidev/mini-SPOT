"""Checks for the paper-style command policy setup: 3D Bezier gait, IK, command environment."""

import mujoco
import numpy as np
import pytest

from sim.bezier3d import LEG_NAMES, Bezier3DGait
from sim.command_env import CommandGaitEnv
from sim.residual_env import FLAT_XML

FLAT = dict(bump_height=(0.0, 0.0), slope_deg=(0.0, 0.0), side_slope_deg=(0.0, 0.0),
            push_force=(0.0, 0.0), randomize=False)


def test_ik_matches_mujoco_kinematics():
    m = mujoco.MjModel.from_xml_path(FLAT_XML)
    d = mujoco.MjData(m)
    gen = Bezier3DGait(m)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    rng = np.random.default_rng(0)
    for i, leg in enumerate(LEG_NAMES):
        for _ in range(20):
            q = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.0, 1.6), rng.uniform(-2.3, -1.0)])
            d.qpos[:] = 0
            d.qpos[3] = 1
            for joint, v in zip(("hip", "thigh", "calf"), q):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint")
                d.qpos[m.jnt_qposadr[jid]] = v
            mujoco.mj_kinematics(m, d)
            site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            foot = d.site_xpos[site] - d.xpos[trunk]
            assert np.allclose(gen.legs[i].fk(q), foot, atol=1e-9)
            assert np.allclose(gen.legs[i].ik(foot), q, atol=1e-6)


def test_observation_and_action_size():
    env = CommandGaitEnv(**FLAT)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 + 3 + 2 + 3 + 14,)
    assert env.action_space.shape == (14,)


def drive(command, seconds=5.0):
    env = CommandGaitEnv(command=command, **dict(FLAT, max_episode_s=seconds))
    obs, _ = env.reset(seed=0)
    vels = []
    while True:
        obs, _, term, trunc, info = env.step(np.zeros(14))
        vels.append(info["vel"])
        if term or trunc:
            return term, np.mean(vels[50:], axis=0)


@pytest.mark.parametrize("command,axis", [
    ((0.15, 0.0, 0.0), 0), ((-0.15, 0.0, 0.0), 0),     # forward / backward
    ((0.0, 0.1, 0.0), 1), ((0.0, -0.1, 0.0), 1),       # strafe left / right
    ((0.0, 0.0, 0.5), 2), ((0.0, 0.0, -0.5), 2),       # turn left / right
])
def test_plain_paper_gait_follows_each_command(command, axis):
    fell, vel = drive(command)
    assert not fell
    assert np.sign(vel[axis]) == np.sign(command[axis])
    assert abs(vel[axis]) > 0.5 * abs(command[axis])
