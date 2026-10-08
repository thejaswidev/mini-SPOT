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


def fit_circle_radius(xy):
    """Least-squares circle fit through the points; returns the radius."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    cx, cy, c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    return np.sqrt(c + cx * cx + cy * cy)


@pytest.mark.parametrize("direction,radius", [("left", 1.0), ("right", 1.0)])
def test_circle_radius(direction, radius):
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    ctrl = GaitController(model, data)
    ctrl.home()

    prev = ctrl.yaw()
    turned = 0.0
    xy = []
    ctrl.circle(direction, radius)
    while data.time < 60.0:
        ctrl.step()
        y = ctrl.yaw()
        turned += wrap(y - prev)
        prev = y
        if data.time > 20.0:                       # skip the start, steering settles first
            xy.append(data.qpos[:2].copy())

    sign = 1 if direction == "left" else -1
    assert sign * turned > np.radians(180)         # really went round, the right way
    assert abs(fit_circle_radius(np.array(xy[::20])) - radius) < 0.15
    assert data.qpos[2] > 0.15                     # did not collapse
