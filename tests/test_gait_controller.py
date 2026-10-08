"""Headless checks for sim/gait_controller.py: turning in place hits the commanded angle."""

import mujoco
import numpy as np
import pytest

from sim.gait_controller import DEFAULT_XML, GaitController, wrap


def run_turn(direction, degrees, timeout_s=30.0, settle_s=1.5):
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    ctrl = GaitController(model, data)
    ctrl.home()

    xy0 = data.qpos[:2].copy()
    prev = ctrl.yaw()
    turned = 0.0
    ctrl.turn(direction, degrees)
    t0, done_at = data.time, None
    while data.time - t0 < timeout_s:
        if ctrl.step() and done_at is None:
            done_at = data.time
        y = ctrl.yaw()
        turned += wrap(y - prev)
        prev = y
        if done_at is not None and data.time - done_at > settle_s:
            break
    return np.degrees(turned), done_at is not None, ctrl, data, xy0


@pytest.mark.parametrize("direction,degrees", [("left", 90), ("right", 90), ("left", 45)])
def test_turn_reaches_angle(direction, degrees):
    turned, finished, ctrl, data, xy0 = run_turn(direction, degrees)
    sign = 1 if direction == "left" else -1
    assert finished, "turn never finished"
    assert ctrl.mode == "stand"
    assert abs(turned - sign * degrees) < 5.0
    assert np.linalg.norm(data.qpos[:2] - xy0) < 0.15     # turned on the spot
    assert data.qpos[2] > 0.15                            # did not collapse
