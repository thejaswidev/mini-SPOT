"""
QUADRUPED BIOMIMETIC LOCOMOTION — MuJoCo, interactive
Multi-gait system: lateral walk, diagonal trot, turn-in-place by an angle.

Run (from the repo root):
    Linux / Docker :  python sim/gait_controller.py
    macOS          :  mjpython sim/gait_controller.py   (the viewer needs mjpython on Mac)

Then type commands in the terminal:
    walk | trot                 walk forward continuously
    turn left [deg]             rotate in place by deg (default 90) then stand
    turn right [deg]
    stop | home                 stand still
    params                      show gait params
    set <param> <value>         shoulder_home | shoulder_sweep | knee_home | knee_lift
                                | cycle_period | turn_period

Turning: HAA (sideways hip) joints are locked, so the robot turns like a tank:
the legs on one side step backward while the other side steps forward. The
trunk yaw is read from the free joint and the turn stops once the requested
angle is reached.

ctrl layout: [FR_sh, FR_kn, FL_sh, FL_kn, RR_sh, RR_kn, RL_sh, RL_kn]
Shoulder: fwd = home - sweep (all legs). Knee: -0.916 extended ... -2.44 folded.
"""

import os
import select
import sys
import time

import mujoco
import numpy as np

# allow both `python sim/gait_controller.py` and `python -m sim.gait_controller`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.bezier_helpers import (  # noqa: E402
    KNEE_MAX,
    KNEE_MIN,
    LEGS,
    TROT_OFFSETS,
    WALK_OFFSETS,
    compute_leg,
    lerp,
    smoothstep,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "assets", "scene.xml")

# ================================================================
#  TUNING BENCH
# ================================================================
CONTROL_FREQUENCY = 50.0     # Hz, gait target update rate (physics runs at 1/timestep)
TRANSITION_TIME   = 0.3      # s, smooth blend into / out of a gait

GAIT_PARAMS = {
    "shoulder_home":  0.687,  # rad
    "shoulder_sweep": 0.35,   # rad, stride length
    "knee_home":     -1.0,    # rad, stance knee
    "knee_lift":     -1.3,    # rad, swing knee
    "cycle_period":   1.4,    # s per gait cycle when walking (lower = faster)
    "turn_period":    1.0,    # s per gait cycle when turning
}
PARAM_BOUNDS = {
    "shoulder_home":  (0.0, np.pi),
    "shoulder_sweep": (0.0, 1.0),
    "knee_home":      (KNEE_MIN, KNEE_MAX),
    "knee_lift":      (KNEE_MIN, KNEE_MAX),
    "cycle_period":   (0.3, 5.0),
    "turn_period":    (0.3, 5.0),
}

# gait name -> (phase offsets FR, FL, RR, RL ; swing fraction)
GAITS = {
    "walk": (WALK_OFFSETS, 0.33),
    "trot": (TROT_OFFSETS, 0.45),
    "turn": (TROT_OFFSETS, 0.45),   # diagonal pairs: turns on the spot with least drift
}

DEFAULT_TURN_DEG = 90.0
TURN_TOLERANCE_DEG = 2.0     # stop this close to the target (the body coasts the rest)


def yaw_of(quat):
    """Heading (rad) from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ================================================================
#  CONTROLLER
# ================================================================
class GaitController:
    """
    Modes: "stand", "walk", "trot", "turn".
    Call step() once per physics step; gait targets are updated at CONTROL_FREQUENCY.
    """

    def __init__(self, model, data, params=None):
        self.model = model
        self.data = data
        self.p = dict(GAIT_PARAMS if params is None else params)
        self.control_period = 1.0 / CONTROL_FREQUENCY

        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root_qpos = model.jnt_qposadr[free[0]]

        self.mode = "stand"
        self.phase = 0.0
        self.turn_dir = 0          # +1 left (CCW), -1 right (CW)
        self.turn_target = 0.0     # rad, absolute amount still to turn
        self.turned = 0.0          # rad, signed yaw accumulated in this turn
        self.prev_yaw = 0.0
        self.last_control = -np.inf
        # blend from the pose held at the start of a mode change to the new targets
        self.blend_from = None
        self.blend_start = 0.0

    # -------------------------------------------------- state helpers
    def yaw(self):
        return yaw_of(self.data.qpos[self.root_qpos + 3:self.root_qpos + 7])

    def set_param(self, name, value):
        if name not in PARAM_BOUNDS:
            raise ValueError(f"Unknown gait parameter: {name} (choose from {', '.join(PARAM_BOUNDS)})")
        lo, hi = PARAM_BOUNDS[name]
        self.p[name] = float(np.clip(value, lo, hi))

    def format_params(self):
        return ", ".join(f"{k}={v:.4f}" for k, v in self.p.items())

    # -------------------------------------------------- commands
    def home(self, settle_steps=500):
        """Snap to the standing pose and let the sim settle (used once at start)."""
        self.mode = "stand"
        self._write_targets(self._stand_targets())
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def stand(self):
        self._change_mode("stand")

    def walk(self, gait="walk"):
        self._change_mode(gait)

    def turn(self, direction, degrees=DEFAULT_TURN_DEG):
        """direction: "left" | "right". Rotates in place by `degrees`, then stands."""
        self.turn_dir = 1 if direction == "left" else -1
        self.turn_target = np.radians(abs(degrees))
        self.turned = 0.0
        self.prev_yaw = self.yaw()
        self._change_mode("turn")

    def _change_mode(self, mode):
        self.blend_from = self._current_ctrl()
        self.blend_start = self.data.time
        self.mode = mode
        self.phase = 0.0
        self.last_control = -np.inf     # update targets on the very next step

    # -------------------------------------------------- gait maths
    def _stand_targets(self):
        return [(self.p["shoulder_home"], self.p["knee_home"])] * len(LEGS)

    def _gait_targets(self):
        offsets, swing_frac = GAITS[self.mode]
        targets = []
        for i, leg in enumerate(LEGS):
            sweep = self.p["shoulder_sweep"]
            if self.mode == "turn":
                # turn left: left legs step backward, right legs forward (and vice versa).
                # A negative sweep makes compute_leg swap forward and back.
                backward = leg["left"] if self.turn_dir > 0 else not leg["left"]
                if backward:
                    sweep = -sweep
            ph = (self.phase + offsets[i]) % 1.0
            targets.append(compute_leg(self.p, sweep, ph, swing_frac))
        return targets

    def _current_ctrl(self):
        return [(self.data.ctrl[leg["sh"]], self.data.ctrl[leg["kn"]]) for leg in LEGS]

    def _write_targets(self, targets):
        for leg, (sh, kn) in zip(LEGS, targets):
            self.data.ctrl[leg["sh"]] = sh
            self.data.ctrl[leg["kn"]] = kn

    # -------------------------------------------------- per-step update
    def step(self):
        """Advance one physics step. Returns a message when a turn finishes, else None."""
        msg = None
        if self.data.time - self.last_control >= self.control_period:
            dt = self.control_period if np.isfinite(self.last_control) else 0.0
            self.last_control = self.data.time

            if self.mode == "turn":
                y = self.yaw()
                self.turned += wrap(y - self.prev_yaw)
                self.prev_yaw = y
                if abs(self.turned) >= self.turn_target - np.radians(TURN_TOLERANCE_DEG):
                    msg = f">> Turn done: rotated {np.degrees(self.turned):+.1f} deg"
                    self._change_mode("stand")

            if self.mode == "stand":
                targets = self._stand_targets()
            else:
                period = self.p["turn_period"] if self.mode == "turn" else self.p["cycle_period"]
                self.phase = (self.phase + dt / period) % 1.0
                targets = self._gait_targets()

            # smooth hand-over between modes
            if self.blend_from is not None:
                t = smoothstep((self.data.time - self.blend_start) / TRANSITION_TIME)
                targets = [(lerp(a[0], b[0], t), lerp(a[1], b[1], t))
                           for a, b in zip(self.blend_from, targets)]
                if t >= 1.0:
                    self.blend_from = None
            self._write_targets(targets)

        mujoco.mj_step(self.model, self.data)
        return msg


# ================================================================
#  COMMAND LINE
# ================================================================
HELP = """  Commands (type in terminal, then Enter):
    walk | trot                 walk forward
    turn left [deg]             rotate in place (default 90 deg), then stand
    turn right [deg]
    stop | home                 stand still
    params                      show gait params
    set <param> <value>         e.g. set shoulder_sweep 0.35"""


def handle_command(ctrl, raw_cmd):
    parts = raw_cmd.strip().lower().split()
    if not parts:
        return
    cmd = parts[0]

    if cmd in ("walk", "trot"):
        print(f">> {cmd.upper()} active")
        ctrl.walk(cmd)

    elif cmd == "turn" and len(parts) >= 2 and parts[1] in ("left", "right"):
        try:
            deg = float(parts[2]) if len(parts) >= 3 else DEFAULT_TURN_DEG
        except ValueError:
            print(">> Usage: turn left|right [degrees]")
            return
        print(f">> TURN {parts[1].upper()} {deg:.0f} deg")
        ctrl.turn(parts[1], deg)

    elif cmd in ("stop", "home"):
        ctrl.stand()
        print(">> Standby")

    elif cmd == "params":
        print(">> Gait params:", ctrl.format_params())

    elif cmd == "set" and len(parts) == 3:
        try:
            ctrl.set_param(parts[1], float(parts[2]))
            print(f">> Set {parts[1]} = {ctrl.p[parts[1]]:.4f}")
        except ValueError as exc:
            print(f">> {exc}")

    else:
        print(">> Unknown command\n" + HELP)


def main():
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    ctrl = GaitController(model, data)
    ctrl.home()

    print("=" * 50)
    print("  Biomimetic Quadruped — MuJoCo")
    print(HELP)
    print(f"    Current: {ctrl.format_params()}")
    print("=" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall0, sim0 = time.time(), data.time
        while viewer.is_running():
            # non-blocking keyboard input (Linux / macOS)
            if select.select([sys.stdin], [], [], 0)[0]:
                handle_command(ctrl, sys.stdin.readline())

            # step physics until it catches up with the wall clock (real-time)
            target_sim = sim0 + (time.time() - wall0)
            for _ in range(100):           # cap, so a slow PC runs in slow-motion instead of freezing
                if data.time >= target_sim:
                    break
                msg = ctrl.step()
                if msg:
                    print(msg)
            else:
                wall0, sim0 = time.time(), data.time
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()
