"""
Bezier Curve Generator (Gamma) from
  "Dynamics and Domain Randomized Gait Modulation with Bezier Curves
   for Sim-to-Real Legged Locomotion" (D^2-GMBC), Rahme et al., arXiv 2010.12070
  Algorithm 3 + Appendix A. The open-loop generator is implemented here,
  along with an optional MuJoCo inverse-kinematics adapter for Mini-SPOT.

=====================================================================
 WHAT GOES IN / WHAT COMES OUT  (summary)
=====================================================================
 INPUTS
   t        : time in seconds                                   (clock)
   rho      : trajectory heading angle in [-pi/2, pi/2]  [rad]  (command zeta)
   yaw_rate : yaw command omega_bar                             (command zeta)
   L_span   : half stride length [m]                            (command zeta)
   psi      : foot clearance height above ground [m]            <-- SET BY RL
   delta    : virtual ground penetration depth   [m]            <-- SET BY RL
   gait     : "trot" or "walk"                                  (fixed choice)

 OUTPUT
   f_xyz : (4, 3) numpy array, rows = [FL, FR, BL, BR], cols = [x, y, z]
           body-relative foot positions [m].
   ctrl  : model.nu target angles from MiniSpotIK, in the loaded model's
           actuator order (supports models with or without HAA joints).
           (In the full D2-GMBC system the policy residual delta_f_xyz would be
            added to this: f_final = f_xyz + delta_f_xyz.  Not done here.)

 WHICH PARAMETERS RL CHANGES (exactly 2 -> beta = [psi, delta])
   psi   : clearance height  -> scales the swing Bezier control-point heights
   delta : penetration depth -> depth of the sinusoidal stance dip
   Everything else (L_span, rho, yaw_rate, gait phase lags, T_swing, stand
   pose, Bezier control-point shape) is fixed / commanded, NOT learned.
   (The paper's policy also outputs 12 foot residuals delta_f_xyz = 4 legs x 3;
    these are a separate policy output and are not part of the generator.)
=====================================================================
"""

import math
import os
import time
import numpy as np

LEGS = ["FL", "FR", "BL", "BR"]  # row order of the 4x3 output

MINISPOT_LEG_ROWS = {
    "FR": 1,
    "FL": 0,
    "RR": 3,
    "RL": 2,
}


# ---------------------------------------------------------------------------
# GAIT CHOICE: trot or walk  (relative phase lag of each leg w.r.t. FL)
# ---------------------------------------------------------------------------
# Fractions of the stride period T_stride, order [FL, FR, BL, BR].
GAITS = {
    # TROT  - Eq. (16) of the paper. Diagonal pairs (FL+BR) and (FR+BL) move
    #         together. Fig. 8: T_stance == T_swing.
    "trot": dict(phase_lag=[0.0, 0.5, 0.5, 0.0], stance_to_swing=1.0),
    # WALK  - NOT in the paper (it only specifies trot). This is& the standard
    #         lateral-sequence walk (FL -> BR -> FR -> BL), added by me. A walk
    #         needs >= 3 feet on the ground, so T_stance = 3 * T_swing.
    "walk": dict(phase_lag=[0.0, 0.5, 0.75, 0.25], stance_to_swing=3.0),
}

# ---------------------------------------------------------------------------
# FIXED (non-learned) constants
# ---------------------------------------------------------------------------
T_SWING = 0.2  # [s] swing duration, "determined empirically" in the paper

# Rest foot position of each leg in the BODY frame, f_xyz^stand.
# PLACEHOLDER numbers (robot specific!) - replace with your robot's stand pose.
F_STAND = np.array([
    [+0.10, +0.08, -0.20],  # FL
    [+0.10, -0.08, -0.20],  # FR
    [-0.10, +0.08, -0.20],  # BL
    [-0.10, -0.08, -0.20],  # BR
])


# ---------------------------------------------------------------------------
# RL-parameter helper: policy output in [-1, 1] -> (psi, delta)
# ---------------------------------------------------------------------------
# The paper's linear policy outputs values in [-1, 1] that are remapped to a
# bounded range. The bounds below are PLACEHOLDERS, tune for your robot.
PSI_RANGE = (0.01, 0.08)     # [m] clearance height bounds
DELTA_RANGE = (0.000, 0.01)  # [m] penetration depth bounds


def beta_from_policy(a_psi: float, a_delta: float):
    """Map the 2 RL outputs in [-1,1] to beta = (psi, delta).  <-- RL PARAMS"""
    def remap(a, lo, hi):
        a = float(np.clip(a, -1.0, 1.0))
        return lo + (a + 1.0) * 0.5 * (hi - lo)
    return remap(a_psi, *PSI_RANGE), remap(a_delta, *DELTA_RANGE)


# ---------------------------------------------------------------------------
# Eq. (14) + Table V: Bezier curve for the SWING phase
# ---------------------------------------------------------------------------
def _bernstein(n: int, k: int, s: float) -> float:
    # B_k^n(s) = C(n,k) (1-s)^(n-k) s^k   (paper's Eq. 14 lost the s^k exponent
    # in typesetting; this is the standard Bernstein polynomial)
    return math.comb(n, k) * (1.0 - s) ** (n - k) * s ** k


def _control_points(tau: float, psi: float) -> np.ndarray:
    """12 control points c_k(tau, psi) in the (q, z) plane, Table V.
    tau = step length, psi = clearance height  (psi is RL-controlled)."""
    return np.array([
        (-tau,        0.0),        # c0
        (-1.4 * tau,  0.0),        # c1
        (-1.5 * tau,  0.9 * psi),  # c2
        (-1.5 * tau,  0.9 * psi),  # c3
        (-1.5 * tau,  0.9 * psi),  # c4
        (0.0,         0.9 * psi),  # c5
        (0.0,         0.9 * psi),  # c6
        (0.0,         1.1 * psi),  # c7
        (+1.5 * tau,  1.1 * psi),  # c8  (sign: see note below)
        (+1.5 * tau,  1.1 * psi),  # c9
        (+1.4 * tau,  0.0),        # c10
        (+tau,        0.0),        # c11
    ])
    # NOTE: the extracted Table V shows negative q for c8..c10. The swing has to
    # travel from -tau (c0) to +tau (c11), so the symmetric positive values are
    # used here (matches the original MIT-Cheetah curve of Hyun et al.).


def _swing(s_swing: float, tau: float, psi: float):
    """Swing: Bezier curve, s_swing in [0,1] -> (q, z)."""
    pts = _control_points(tau, psi)
    n = len(pts) - 1
    q = z = 0.0
    for k in range(n + 1):
        b = _bernstein(n, k, s_swing)
        q += b * pts[k, 0]
        z += b * pts[k, 1]
    return q, z


def _stance(s_stance: float, tau: float, delta: float):
    """Stance: sinusoidal dip, s_stance in [0,1) -> (q, z).
    Foot slides backwards from +tau to -tau while dipping by `delta`
    (delta is RL-controlled).  z = 0 at both ends so it joins the swing curve."""
    u = 1.0 - 2.0 * s_stance                 # +1 -> -1 over the stance
    q = tau * u
    z = -delta * math.cos(0.5 * math.pi * u)  # 0 at u=+-1, -delta at u=0
    return q, z


def trajectory_generator(S: float, tau: float, psi: float, delta: float):
    """Upsilon(S, tau): phase S in [0,2) -> planar point (q, z).
       S in [0,1): stance (sine curve)   S in [1,2): swing (Bezier curve)."""
    if S < 1.0:
        return _stance(S, tau, delta)
    return _swing(S - 1.0, tau, psi)


# ---------------------------------------------------------------------------
# Eq. (15)-(17): per-leg phase S_l(t)
# ---------------------------------------------------------------------------
def leg_phase(t: float, leg_idx: int, gait: str) -> float:
    cfg = GAITS[gait]
    T_swing = T_SWING
    T_stance = cfg["stance_to_swing"] * T_swing
    T_stride = T_stance + T_swing
    # Eq. (15): shift the clock by the leg's phase lag (fraction of stride).
    t_l = t - cfg["phase_lag"][leg_idx] * T_stride
    # Eq. (17) written equivalently with a modulo (periodic, no contact sensing):
    u = t_l % T_stride
    if u < T_stance:
        return u / T_stance                       # stance:  S in [0, 1)
    return 1.0 + (u - T_stance) / T_swing         # swing:   S in [1, 2)


# ---------------------------------------------------------------------------
# Algorithm 3: the generator itself
# ---------------------------------------------------------------------------
class BezierGaitGenerator:
    """Gamma(S(t), zeta, beta) -> f_xyz  (4x3).  Stateful only because the yaw
    term (Eq. 6) uses the previous foot position."""

    def __init__(self, gait: str = "trot", f_stand: np.ndarray = F_STAND):
        assert gait in GAITS, f"gait must be one of {list(GAITS)}"
        self.gait = gait
        self.f_stand = np.asarray(f_stand, dtype=float)
        self.f_prev = self.f_stand.copy()   # g_xyz(0) = 0  (Eq. 6)

    def step(self, t, rho, yaw_rate, L_span, psi, delta):
        """
        INPUTS
          t         : time [s]
          rho       : heading angle of the translation   [rad]   (command)
          yaw_rate  : omega_bar, yaw step magnitude/sign         (command)
          L_span    : half stride length [m]                     (command)
          psi       : clearance height  [m]     <-- RL PARAM #1  (beta[0])
          delta     : penetration depth [m]     <-- RL PARAM #2  (beta[1])
        OUTPUT
          f_xyz (4,3): rows FL,FR,BL,BR; cols x,y,z; body-relative foot targets
        """
        f_xyz = np.zeros((4, 3))
        for l in range(4):
            # 1: map t -> leg phase S_l(t)     (Eq. 15-17, trot or walk lags)
            S = leg_phase(t, l, self.gait)

            # 2: planar translation curve  (f_q^tr, f_z^tr) = Upsilon(S, L_span)
            fq_tr, fz_tr = trajectory_generator(S, L_span, psi, delta)
            # 3: planar yaw curve          (f_q^yaw, f_z^yaw) = Upsilon(S, omega_bar)
            fq_yaw, fz_yaw = trajectory_generator(S, yaw_rate, psi, delta)

            # 4: lift translation curve to 3D by rotating by rho  (Eq. 5)
            f_tr = np.array([fq_tr * math.cos(rho), fq_tr * math.sin(rho), fz_tr])

            # 5: yaw direction for this leg (Eq. 6-10), 4-wheel-steering style
            g = self.f_prev[l] - self.f_stand[l]            # Eq. 6
            g_ang = math.atan2(g[1], g[0])                  # Eq. 7-8 (angle)
            fx_s, fy_s = self.f_stand[l, 0], self.f_stand[l, 1]
            sign = +1.0 if LEGS[l] in ("FR", "BL") else -1.0  # Eq. 10
            phi_stand = sign * math.atan2(fy_s, fx_s)
            phi_arc = g_ang + phi_stand + 0.5 * math.pi     # Eq. 9

            # 6: yaw curve in 3D (Eq. 11)
            # NOTE: z of Upsilon depends only on (S, psi, delta), not on the step
            # length, so fz_yaw == fz_tr. Summing both (literal Eq. 4) would
            # double the foot lift and penetration, so the yaw term carries
            # NO z component here (z comes from the translation curve only).
            f_yaw = np.array([fq_yaw * math.cos(phi_arc),
                              fq_yaw * math.sin(phi_arc),
                              0.0])

            # 7: f = translation + yaw + rest pose   (Eq. 4)
            f_xyz[l] = f_tr + f_yaw + self.f_stand[l]

        self.f_prev = f_xyz.copy()
        # 8: return to the robot for IK.  (Full system: + delta_f_xyz from RL.)
        return f_xyz

    def step_mujoco(self, model, data, t, rho, yaw_rate, L_span, psi, delta):
        """Return Mini-SPOT actuator targets for this time step.

        The result is a ``(model.nu,)`` array ordered like the loaded model's
        actuators, ready to assign to ``data.ctrl``. Models with HAA joints use
        full 3D IK; the current Mini-SPOT model without HAA uses sagittal-plane
        IK and therefore cannot produce lateral foot motion.
        """
        controller = getattr(self, "_minispot_ik", None)
        if controller is None or controller.model is not model:
            controller = MiniSpotIK(model)
            self._minispot_ik = controller
        foot_targets = self.step(t, rho, yaw_rate, L_span, psi, delta)
        return controller.targets(foot_targets, data)


class MiniSpotIK:
    """Convert body-frame foot targets to Mini-SPOT joint position controls."""

    _LEG_JOINT_SUFFIXES = {
        "haa": ("hip_joint", "hip"),
        "hfe": ("thigh_joint", "thigh"),
        "kfe": ("calf_joint", "calf"),
    }

    def __init__(self, model, *, max_iterations=60, damping=0.04,
                 position_tolerance=1e-4, max_position_error=0.02):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo is required to convert Mini-SPOT foot targets to joint controls."
            ) from exc

        self.mujoco = mujoco
        self.model = model
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.position_tolerance = float(position_tolerance)
        self.max_position_error = float(max_position_error)
        if self.max_iterations <= 0 or self.damping <= 0.0:
            raise ValueError("max_iterations and damping must be positive.")
        if self.position_tolerance <= 0.0 or self.max_position_error <= 0.0:
            raise ValueError("IK error tolerances must be positive.")

        self.ik_data = mujoco.MjData(model)
        self.trunk_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.legs = {}
        for leg in MINISPOT_LEG_ROWS:
            joint_ids = {}
            actuator_ids = {}
            for joint_name, actuator_name in self._LEG_JOINT_SUFFIXES.values():
                joint_id = self._optional_name_id(
                    mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint_name}"
                )
                actuator_id = self._optional_name_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{actuator_name}"
                )
                if (joint_id is None) != (actuator_id is None):
                    raise ValueError(
                        f"{leg} must have both the {joint_name} joint and its "
                        f"{actuator_name} actuator, or neither."
                    )
                if joint_id is not None:
                    joint_ids[joint_name] = joint_id
                    actuator_ids[actuator_name] = actuator_id

            if "thigh_joint" not in joint_ids or "calf_joint" not in joint_ids:
                raise ValueError(
                    f"{leg} needs the thigh and calf joints and actuators."
                )
            site_id = self._name_id(
                mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"
            )
            dof_ids = {
                name: int(model.jnt_dofadr[joint_id])
                for name, joint_id in joint_ids.items()
            }
            qpos_ids = {
                name: int(model.jnt_qposadr[joint_id])
                for name, joint_id in joint_ids.items()
            }

            for joint_name, joint_id in joint_ids.items():
                actuator_name = self._LEG_JOINT_SUFFIXES[
                    next(
                        key
                        for key, (suffix, _) in self._LEG_JOINT_SUFFIXES.items()
                        if suffix == joint_name
                    )
                ][1]
                actuator_id = actuator_ids[actuator_name]
                if model.actuator_trnid[actuator_id, 0] != joint_id:
                    raise ValueError(
                        f"Actuator for {leg} does not control its expected joint."
                    )
                if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"{leg} IK expects hinge joints.")

            self.legs[leg] = {
                "joint_ids": joint_ids,
                "actuator_ids": actuator_ids,
                "dof_ids": dof_ids,
                "qpos_ids": qpos_ids,
                "site_id": site_id,
            }
        self.reference_feet_body = None
        self.has_haa = all("hip_joint" in leg["joint_ids"] for leg in self.legs.values())

    def _name_id(self, object_type, name):
        object_id = self.mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(
                f"Mini-SPOT model is missing '{name}'."
            )
        return int(object_id)

    def _optional_name_id(self, object_type, name):
        object_id = self.mujoco.mj_name2id(self.model, object_type, name)
        return None if object_id < 0 else int(object_id)

    def targets(self, foot_targets, data):
        """Solve IK and return actuator controls in the model's actuator order.

        ``foot_targets`` uses the generator's row order [FL, FR, BL, BR].
        MuJoCo's Mini-SPOT order is resolved by names, so actuator XML order
        does not need to match that row order.
        """
        foot_targets = np.asarray(foot_targets, dtype=float)
        if foot_targets.shape != (4, 3) or not np.isfinite(foot_targets).all():
            raise ValueError("foot_targets must be a finite array with shape (4, 3).")
        if data.qpos.shape != self.ik_data.qpos.shape:
            raise ValueError("The MuJoCo data does not belong to the supplied model.")

        self.ik_data.qpos[:] = data.qpos
        self.ik_data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.ik_data)

        trunk_pos = self.ik_data.xpos[self.trunk_id].copy()
        trunk_rot = self.ik_data.xmat[self.trunk_id].reshape(3, 3).copy()
        if self.reference_feet_body is None:
            self.reference_feet_body = {}
            for leg, info in self.legs.items():
                site_pos = self.ik_data.site_xpos[info["site_id"]]
                self.reference_feet_body[leg] = trunk_rot.T @ (site_pos - trunk_pos)

        controls = np.asarray(data.ctrl, dtype=float).copy()

        for leg, row in MINISPOT_LEG_ROWS.items():
            target_body = (
                self.reference_feet_body[leg] + foot_targets[row] - F_STAND[row]
            )
            target_world = trunk_pos + trunk_rot @ target_body
            info = self.legs[leg]
            jacobian = np.zeros((3, self.model.nv))
            jacobian_rot = np.zeros_like(jacobian)
            dof_names = ["hip_joint", "thigh_joint", "calf_joint"]
            dof_names = [name for name in dof_names if name in info["dof_ids"]]
            dof_ids = [info["dof_ids"][name] for name in dof_names]

            for _ in range(self.max_iterations):
                self.mujoco.mj_forward(self.model, self.ik_data)
                current = self.ik_data.site_xpos[info["site_id"]]
                error = target_world - current
                error_body = trunk_rot.T @ error
                if self.has_haa:
                    active_error = error
                else:
                    active_error = error_body[[0, 2]]
                if np.linalg.norm(active_error) <= self.position_tolerance:
                    break

                self.mujoco.mj_jacSite(
                    self.model, self.ik_data, jacobian, jacobian_rot, info["site_id"]
                )
                jacobian_body = trunk_rot.T @ jacobian
                if not self.has_haa:
                    active_jacobian = jacobian_body[[0, 2]][:, dof_ids]
                else:
                    active_jacobian = jacobian[:, dof_ids]
                system = (
                    active_jacobian @ active_jacobian.T
                    + self.damping**2 * np.eye(active_jacobian.shape[0])
                )
                delta_q = active_jacobian.T @ np.linalg.solve(system, active_error)
                delta_q = np.clip(delta_q, -0.2, 0.2)

                for joint_name, change in zip(dof_names, delta_q):
                    joint_id = info["joint_ids"][joint_name]
                    qpos_id = info["qpos_ids"][joint_name]
                    angle = self.ik_data.qpos[qpos_id] + change
                    if self.model.jnt_limited[joint_id]:
                        lower, upper = self.model.jnt_range[joint_id]
                        angle = np.clip(angle, lower, upper)
                    self.ik_data.qpos[qpos_id] = angle

            self.mujoco.mj_forward(self.model, self.ik_data)
            residual_body = trunk_rot.T @ (
                target_world - self.ik_data.site_xpos[info["site_id"]]
            )
            residual = (
                np.linalg.norm(residual_body)
                if self.has_haa
                else np.linalg.norm(residual_body[[0, 2]])
            )
            if residual > self.max_position_error:
                raise ValueError(
                    f"{leg} foot target is unreachable within joint limits "
                    f"(position error {residual:.4f} m)."
                )

            for joint_name, joint_id in info["joint_ids"].items():
                actuator_name = next(
                    actuator
                    for actuator, suffix in (
                        ("hip", "hip_joint"),
                        ("thigh", "thigh_joint"),
                        ("calf", "calf_joint"),
                    )
                    if suffix == joint_name
                )
                actuator_id = info["actuator_ids"][actuator_name]
                qpos_id = info["qpos_ids"][joint_name]
                angle = self.ik_data.qpos[qpos_id]
                if self.model.actuator_ctrllimited[actuator_id]:
                    lower, upper = self.model.actuator_ctrlrange[actuator_id]
                    angle = np.clip(angle, lower, upper)
                controls[actuator_id] = angle

        return controls


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def run_mujoco_demo(xml_path=None, gait="trot"):
    """Open the Mini-SPOT scene and run the gait generator in a passive viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise ImportError(
            "Install the MuJoCo Python package to run the interactive demo."
        ) from exc

    here = os.path.dirname(os.path.abspath(__file__))
    if xml_path is None:
        xml_path = os.path.join(here, "assets", "scene.xml")

    model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
    data = mujoco.MjData(model)
    generator = BezierGaitGenerator(gait=gait)

    for leg, info in generator_ik_legs(model, mujoco).items():
        for joint_name, angle in (("thigh_joint", 0.85), ("calf_joint", -1.35)):
            joint_id = info.get(joint_name)
            if joint_id is None:
                continue
            qpos_id = model.jnt_qposadr[joint_id]
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                angle = np.clip(angle, lower, upper)
            data.qpos[qpos_id] = angle
        haa_id = info.get("hip_joint")
        if haa_id is not None:
            qpos_id = model.jnt_qposadr[haa_id]
            data.qpos[qpos_id] = 0.0

    mujoco.mj_forward(model, data)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id >= 0:
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    control_period = 1.0 / 50.0
    simulation_steps = max(1, round(control_period / model.opt.timestep))
    psi, delta = 0.045, 0.0025

    print(f"Opening MuJoCo scene: {os.path.abspath(xml_path)}")
    print(f"Gait: {gait}; actuators: {model.nu}; HAA enabled: {model.nu == 12}")
    print("Close the MuJoCo viewer window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            controls = generator.step_mujoco(
                model=model,
                data=data,
                t=data.time,
                rho=0.0,
                yaw_rate=0.0,
                L_span=0.035,
                psi=psi,
                delta=delta,
            )
            data.ctrl[:] = controls
            for _ in range(simulation_steps):
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = control_period - (time.perf_counter() - start)
            if remaining > 0.0:
                time.sleep(remaining)


def generator_ik_legs(model, mujoco):
    """Resolve the Mini-SPOT leg joints for the demo's initial pose."""
    legs = {}
    for leg in ("FR", "FL", "RR", "RL"):
        joints = {}
        for suffix in ("hip_joint", "thigh_joint", "calf_joint"):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{suffix}"
            )
            if joint_id >= 0:
                joints[suffix] = int(joint_id)
        legs[leg] = joints
    return legs


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_mujoco_demo()