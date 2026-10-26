"""
3D Bezier gait from D^2-GMBC (Rahme et al. 2020, arXiv 2010.12070) for Mini-SPOT, with a
closed-form leg IK so it is fast enough to run inside RL training.

  command  (vx, vy, wz)  body-frame m/s and rad/s     -> forward/back, strafe, turn
       |                                                  (one gait for all directions)
       v
  per leg: phase S in [0, 2)  (stance 0-1, swing 1-2, trot timing)
       |
       |  translation curve  (q, z) = Upsilon(S, L_span, psi, delta)   paper Eq. 13, Table V
       |  yaw curve          (q, -) = Upsilon(S, omega_bar, ...)        paper Eq. 4-11
       v
  foot target f_xyz = f_stand + translation (rotated by rho) + yaw (tangent to the turn circle)
       |                                                               (+ RL residual delta_f)
       v
  IK (closed form, per leg): f_xyz -> (hip, thigh, knee) angles  -> MuJoCo position servos

The planar curve Upsilon (12-point Bezier swing + sine stance) is reused from the repo's
3dbezierhelpers.py. psi = foot clearance height, delta = virtual ground penetration depth:
the two curve parameters the paper's policy sets.

Yaw: the paper builds each foot's turning direction from its previous position (Eq. 6-10). Here
the foot moves tangent to the circle through its stand position, which is what those equations
aim for (four-wheel-steering turn about the body centre) and has no state.
"""

import importlib
import os
import sys

import numpy as np
import mujoco

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
trajectory_generator = importlib.import_module("3dbezierhelpers").trajectory_generator

LEG_NAMES = ["FR", "FL", "RR", "RL"]
TROT_LAGS = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}     # diagonal pairs (paper Eq. 16)
STAND_ANGLES = (0.0, 0.687, -1.0)                           # (hip, thigh, knee) standing pose

PSI_DEFAULT = 0.04       # m, swing clearance height
DELTA_DEFAULT = 0.005    # m, stance penetration depth


def _rot_angle(x, z):
    """Angle of a vector in the x-z plane, measured from straight down (-z), positive to -x.
    A rotation about +y by q adds q to this angle (MuJoCo's thigh / knee axis)."""
    return np.arctan2(-x, -z)


class LegIK:
    """Closed-form IK / FK for one 3-DOF Mini-SPOT leg, built from the MuJoCo model.

    Frames: everything in the trunk (body) frame. The hip (HAA) joint turns about x at the hip
    body origin; the thigh (HFE) and knee (KFE) joints turn about y.
    """

    def __init__(self, model, leg):
        def body_pos(name):
            return model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
        self.hip = body_pos(f"{leg}_hip")                 # hip origin in the trunk frame
        thigh, calf, foot = body_pos(f"{leg}_thigh"), body_pos(f"{leg}_calf"), model.site_pos[site]
        self.tx, self.tz = thigh[0], thigh[2]             # thigh pivot (x, z) in the hip frame
        self.d = thigh[1] + calf[1] + foot[1]             # sideways offset of the foot (constant)
        self.l1 = np.array([calf[0], calf[2]])            # thigh pivot -> knee   (x, z)
        self.l2 = np.array([foot[0], foot[2]])            # knee -> foot          (x, z)
        A = self.l1 @ self.l2
        B = self.l1[0] * self.l2[1] - self.l1[1] * self.l2[0]
        self.R, self.gamma = np.hypot(A, B), np.arctan2(B, A)
        self.ll = self.l1 @ self.l1 + self.l2 @ self.l2

    def fk(self, q):
        """(hip, thigh, knee) -> foot position in the trunk frame."""
        q0, q1, q2 = q
        c2, s2 = np.cos(q2), np.sin(q2)
        u = self.l1 + np.array([c2 * self.l2[0] + s2 * self.l2[1], -s2 * self.l2[0] + c2 * self.l2[1]])
        c1, s1 = np.cos(q1), np.sin(q1)
        x = self.tx + c1 * u[0] + s1 * u[1]
        z = self.tz - s1 * u[0] + c1 * u[1]
        c0, s0 = np.cos(q0), np.sin(q0)
        return self.hip + np.array([x, c0 * self.d - s0 * z, s0 * self.d + c0 * z])

    def ik(self, f):
        """Foot position in the trunk frame -> (hip, thigh, knee) angles, knee bent backward."""
        px, py, pz = np.asarray(f, dtype=float) - self.hip
        # hip (about x): the leg plane sits at sideways offset d; find its height zl
        r_yz2 = py * py + pz * pz
        zl = -np.sqrt(max(r_yz2 - self.d * self.d, 1e-9))
        q0 = np.arctan2(py, -pz) - np.arctan2(self.d, -zl)
        # thigh + knee (about y), target in the thigh-pivot frame
        x, z = px - self.tx, zl - self.tz
        k = np.clip((x * x + z * z - self.ll) / (2.0 * self.R), -1.0, 1.0)
        q2 = self.gamma - np.arccos(k)
        c2, s2 = np.cos(q2), np.sin(q2)
        u = self.l1 + np.array([c2 * self.l2[0] + s2 * self.l2[1], -s2 * self.l2[0] + c2 * self.l2[1]])
        q1 = _rot_angle(x, z) - _rot_angle(*u)
        return np.array([q0, q1, q2])


class Bezier3DGait:
    """Command-driven foot trajectories + IK -> actuator targets (paper Algorithm 3)."""

    def __init__(self, model, t_swing=0.2, stance_to_swing=1.0):   # paper: T_swing = 0.2 s
        self.model = model
        self.t_swing = t_swing
        self.t_stance = stance_to_swing * t_swing
        self.t_stride = self.t_swing + self.t_stance
        self.stance_frac = self.t_stance / self.t_stride
        self.legs = [LegIK(model, leg) for leg in LEG_NAMES]
        self.lags = np.array([TROT_LAGS[leg] for leg in LEG_NAMES])
        self.f_stand = np.array([leg.fk(STAND_ANGLES) for leg in self.legs])     # (4, 3)
        # MuJoCo actuator index of (hip, thigh, knee) for each leg
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.act = [(aid(f"{l}_hip"), aid(f"{l}_thigh"), aid(f"{l}_calf")) for l in LEG_NAMES]
        if min(min(a) for a in self.act) < 0:
            raise ValueError("Bezier3DGait needs the 12-actuator model (hip, thigh, calf per leg)")
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    def leg_phase(self, phase, i):
        """Stride phase in [0, 1) -> paper's S in [0, 2): stance 0-1, swing 1-2."""
        u = (phase + self.lags[i]) % 1.0
        if u < self.stance_frac:
            return u / self.stance_frac
        return 1.0 + (u - self.stance_frac) / (1.0 - self.stance_frac)

    def command_to_curve(self, cmd):
        """(vx, vy, wz) -> (rho, L_span, omega_bar) for the planar curves."""
        vx, vy, wz = cmd
        rho = np.arctan2(vy, vx)
        L_span = np.hypot(vx, vy) * self.t_stance / 2.0       # half stride: covered in one stance
        return rho, L_span, wz * self.t_stance / 2.0          # omega_bar in rad (x radius below)

    def foot_targets(self, phase, cmd, psi=PSI_DEFAULT, delta=DELTA_DEFAULT, residual=None):
        rho, L_span, yaw_half = self.command_to_curve(cmd)
        f = self.f_stand.copy()
        for i in range(4):
            S = self.leg_phase(phase, i)
            q_tr, z = trajectory_generator(S, L_span, psi, delta)
            fx, fy = self.f_stand[i, :2]
            radius = np.hypot(fx, fy)
            q_yaw, _ = trajectory_generator(S, yaw_half * radius, psi, delta)
            tangent = np.array([-fy, fx]) / radius            # counter-clockwise turn direction
            f[i, :2] += q_tr * np.array([np.cos(rho), np.sin(rho)]) + q_yaw * tangent
            f[i, 2] += z
        if residual is not None:
            f += residual
        return f

    def joint_targets(self, foot_targets):
        ctrl = np.zeros(self.model.nu)
        for leg, ids, f in zip(self.legs, self.act, foot_targets):
            ctrl[list(ids)] = leg.ik(f)
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def stand_ctrl(self):
        return self.joint_targets(self.f_stand)
