"""
sim/mujoco_env.py

MuJoCo simulation environment for Mini-SPOT.

Adapted to the actual mini_spot.xml model:
  - HAA joints are DISABLED (range locked to 0). 8 active DOF, not 12.
  - Leg naming: FR, FL, RR, RL  (Rear, not Back)
  - Joint naming: {leg}_thigh_joint (HFE), {leg}_calf_joint (KFE)
  - Actuator order: FR_thigh, FR_calf, FL_thigh, FL_calf,
                    RR_thigh, RR_calf, RL_thigh, RL_calf

Physics:  1000 Hz  (timestep = 0.001 s)
Control:   100 Hz  (every 10 physics steps)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os


# ── Actuator ordering ─────────────────────────────────────────────────────────
# Must match the <actuator> block order in mini_spot.xml exactly.
# Index 0 = FR_thigh (HFE), index 1 = FR_calf (KFE), etc.
ACTUATOR_NAMES = [
    "FR_thigh", "FR_calf",
    "FL_thigh", "FL_calf",
    "RR_thigh", "RR_calf",
    "RL_thigh", "RL_calf",
]

# Human-readable leg/joint mapping for the rest of the codebase
# LEGS[leg][joint] → index into the 8-element control vector
LEGS = {
    "FR": {"HFE": 0, "KFE": 1},
    "FL": {"HFE": 2, "KFE": 3},
    "RR": {"HFE": 4, "KFE": 5},
    "RL": {"HFE": 6, "KFE": 7},
}

# Sensor names (must match <sensor> block in mini_spot.xml)
SENSOR_NAMES = {
    "imu_accel":     "imu_accel",
    "imu_gyro":      "imu_gyro",
    "FR_foot_force": "FR_foot_force",
    "FL_foot_force": "FL_foot_force",
    "RR_foot_force": "RR_foot_force",
    "RL_foot_force": "RL_foot_force",
}

# Joint limits read from XML defaults (radians) — for reference and safety clamping
JOINT_LIMITS = {
    "HFE": (-0.785, 3.14),
    "KFE": (-2.44, -0.916),
}


class MiniSpotEnv:
    """
    Thin wrapper around MuJoCo for Mini-SPOT.

    The robot has 8 active actuators (HAA locked, only HFE + KFE per leg).
    All control vectors are length-8, ordered FR → FL → RR → RL.

    Usage:
        env = MiniSpotEnv()
        obs = env.reset()
        for _ in range(1000):
            angles = np.array([0.687, -2.17, 0.687, -2.17,
                                0.687, -2.17, 0.687, -2.17])  # start keyframe
            env.step(angles)
            obs = env.get_obs()
    """

    PHYSICS_HZ     = 1000
    CONTROL_HZ     = 100
    STEPS_PER_CTRL = PHYSICS_HZ // CONTROL_HZ  # 10

    def __init__(self, xml_path: str = None):
        if xml_path is None:
            here     = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(here, "..", "assets", "scene.xml")

        self.xml_path = os.path.abspath(xml_path)
        print(f"[MiniSpotEnv] Loading: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data  = mujoco.MjData(self.model)

        # Cache actuator IDs once — avoids repeated string lookups in the loop
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]

        # Validate — catch typos in XML early
        for name, aid in zip(ACTUATOR_NAMES, self._actuator_ids):
            assert aid >= 0, f"Actuator '{name}' not found in model. Check XML."

        # Cache sensor addresses
        self._sensor_adr  = {}
        self._sensor_size = {}
        for key, sname in SENSOR_NAMES.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            assert sid >= 0, f"Sensor '{sname}' not found. Did you add the <sensor> block?"
            self._sensor_adr[key]  = self.model.sensor_adr[sid]
            self._sensor_size[key] = self.model.sensor_dim[sid]

        self._print_model_info()

    # ── Model summary ─────────────────────────────────────────────────────────

    def _print_model_info(self):
        print(f"  Bodies    : {self.model.nbody}")
        print(f"  Joints    : {self.model.njnt}")
        print(f"  Actuators : {self.model.nu}  (8 active, HAA locked)")
        print(f"  Sensors   : {self.model.nsensor}")
        print(f"  DOF (nv)  : {self.model.nv}")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, use_keyframe: str = "start"):
        """
        Reset to a named keyframe defined in the XML, or to all-zeros.

        Args:
            use_keyframe: "start" (crouched), "home" (extended), or None for zeros.
        """
        mujoco.mj_resetData(self.model, self.data)

        if use_keyframe is not None:
            kid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, use_keyframe
            )
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
                print(f"[MiniSpotEnv] Reset to keyframe '{use_keyframe}'.")
            else:
                print(f"[MiniSpotEnv] Keyframe '{use_keyframe}' not found, using zeros.")

        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, joint_angles: np.ndarray):
        """
        Apply joint angle targets and advance one control step (10ms).

        Args:
            joint_angles: np.ndarray shape (8,)
                [FR_HFE, FR_KFE, FL_HFE, FL_KFE, RR_HFE, RR_KFE, RL_HFE, RL_KFE]
                All values in radians.
        """
        assert joint_angles.shape == (8,), \
            f"Expected (8,) — 8 active DOF (HAA locked). Got {joint_angles.shape}"

        # Safety clamp to joint limits before sending to actuators
        clamped = self._clamp_to_limits(joint_angles)

        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = clamped[i]

        for _ in range(self.STEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)

    def _clamp_to_limits(self, angles: np.ndarray) -> np.ndarray:
        """Clamp control targets to joint limits. Prevents actuator windup."""
        out = angles.copy()
        for i in range(8):
            joint_type = "HFE" if i % 2 == 0 else "KFE"
            lo, hi = JOINT_LIMITS[joint_type]
            out[i] = np.clip(out[i], lo, hi)
        return out

    # ── Observations ──────────────────────────────────────────────────────────

    def get_obs(self) -> dict:
        """
        Return robot state and sensor readings.

        Returns dict:
          'time'          : float — simulation time (s)
          'qpos'          : (nq,) — full joint position vector
          'qvel'          : (nv,) — full joint velocity vector
          'joint_angles'  : (8,) — active joint angles [FR_HFE, FR_KFE, ...]
          'imu_accel'     : (3,) — trunk accelerometer, body frame (m/s²)
          'imu_gyro'      : (3,) — trunk gyroscope, body frame (rad/s)
          'foot_forces'   : dict {FR, FL, RR, RL} → (3,) force (N)
          'foot_contacts' : dict {FR, FL, RR, RL} → bool
          'torso_pos'     : (3,) — trunk position, world frame (m)
          'torso_quat'    : (4,) — trunk quaternion [w,x,y,z], world frame
        """
        sd = self.data.sensordata

        def read_sensor(key: str) -> np.ndarray:
            adr  = self._sensor_adr[key]
            size = self._sensor_size[key]
            return sd[adr : adr + size].copy()

        foot_forces = {
            leg: read_sensor(f"{leg}_foot_force")
            for leg in ["FR", "FL", "RR", "RL"]
        }
        # Contact: foot is on the ground if net force > 1 N
        foot_contacts = {
            leg: float(np.linalg.norm(f)) > 1.0
            for leg, f in foot_forces.items()
        }

        # Freejoint is always the first joint → qpos[0:7]
        torso_pos  = self.data.qpos[0:3].copy()
        torso_quat = self.data.qpos[3:7].copy()  # MuJoCo: [w, x, y, z]

        # Active joint angles — skip the freejoint DOFs (first 7 in qpos)
        # Then skip any locked HAA joints (they exist as joints but with range 0 0)
        # Easiest: read directly from actuator state
        joint_angles = np.array([
            self.data.ctrl[aid] for aid in self._actuator_ids
        ])

        return {
            "time":          self.data.time,
            "qpos":          self.data.qpos.copy(),
            "qvel":          self.data.qvel.copy(),
            "joint_angles":  joint_angles,
            "imu_accel":     read_sensor("imu_accel"),
            "imu_gyro":      read_sensor("imu_gyro"),
            "foot_forces":   foot_forces,
            "foot_contacts": foot_contacts,
            "torso_pos":     torso_pos,
            "torso_quat":    torso_quat,
        }

    # ── Standing pose helpers ─────────────────────────────────────────────────

    @staticmethod
    def start_pose() -> np.ndarray:
        """
        'start' keyframe from XML: ctrl = [0.687, 0, 0.687, 0, 0.687, 0, 0.687, 0]
        HFE = 0.687 rad (~39°), KFE = 0.0 rad.
        This is the nominal crouched stand pose.
        """
        return np.array([0.687, 0.0,  # FR
                         0.687, 0.0,  # FL
                         0.687, 0.0,  # RR
                         0.687, 0.0]) # RL

    @staticmethod
    def home_pose() -> np.ndarray:
        """
        'home' keyframe from XML: ctrl = [3.14, -2.17, ...]
        More extended pose.
        """
        return np.array([3.14, -2.17,  # FR
                         3.14, -2.17,  # FL
                         3.14, -2.17,  # RR
                         3.14, -2.17]) # RL

    # ── Viewer ────────────────────────────────────────────────────────────────

    def run_viewer(self, controller_fn=None, keyframe: str = "start"):
        """
        Launch the interactive MuJoCo viewer.

        Args:
            controller_fn : callable(env) → np.ndarray(8,), or None to hold start pose.
            keyframe      : which pose to reset to before starting.
        """
        print("[MiniSpotEnv] Launching viewer. Close window to exit.")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.reset(use_keyframe=keyframe)

            while viewer.is_running():
                t0 = time.time()

                angles = controller_fn(self) if controller_fn else self.start_pose()
                self.step(angles)
                viewer.sync()

                # Real-time pacing
                dt = time.time() - t0
                sleep = (1.0 / self.CONTROL_HZ) - dt
                if sleep > 0:
                    time.sleep(sleep)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = MiniSpotEnv()
    env.run_viewer()
