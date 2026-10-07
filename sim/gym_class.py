"""
Gymnasium environment: RL tunes Bezier gait parameters for a MuJoCo quadruped.

  action  (normalised [-1, 1]^6) -> gait parameters
      0 shoulder_home      rad
      1 sweep_left         rad   (left-side legs  FL, RL)
      2 sweep_right        rad   (right-side legs FR, RR)   -> L/R difference = steering
      3 knee_home          rad
      4 knee_lift          rad
      5 cycle_period       s     (the swing fraction is kept fixed; see swing_fraction arg)

  ONE ACTION PER EPISODE: the policy picks the gait parameters once. step(action)
  then runs the whole episode (max_episode_s) with those parameters held fixed
  and returns the summed reward. Only the phase advances inside the episode.
  The Bezier generator runs inside the env at CONTROL_FREQUENCY; `policy_dt` is
  now just the chunk length used for reward averaging and termination checks.

  obs (34): joint pos(8) | joint vel(8) | projected gravity(3) | body lin vel(3)
            | gyro(3) | height(1) | phase sin/cos(2) | current params, normalised(6)
            
"""

import os
# import faulthandler

# # Print a Python traceback if we hit a native crash (segfault) instead of dying silently.
# faulthandler.enable()

# # torch (pulled in by stable-baselines3) and MuJoCo both ship native libs that use
# # OpenMP. With multiple OpenMP runtimes / thread pools in one process the Linux
# # container can segfault right after "Using cpu device". One thread is plenty for
# # this tiny MLP and the sim is single-threaded anyway.
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .bezier_helpers import (
    LEGS,
    PARAM_HIGH,
    PARAM_LOW,
    PARAM_NAMES,
    TROT_OFFSETS,
    WALK_OFFSETS,
    compute_leg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "assets", "scene.xml")


# ---------------------------------------------------------------- environment
class QuadrupedBezierEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path=DEFAULT_XML,        # resolved relative to this file, not the cwd
        render_mode="human",  # "human" | None
        gait="walk",                 # "walk" | "trot"
        swing_fraction=None,         # default 0.33 walk / 0.45 trot
        control_freq=50.0,           # Hz, Bezier target update rate
        policy_dt=0.2,               # s per reward/termination chunk inside an episode
        max_episode_s=20.0,
        target_speed=0.3,            # m/s along body forward axis
        forward_axis=(1.0, 0.0, 0.0),  # body-frame forward direction  (CHECK for your model!)
        reward_weights=None,
        seed=None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.offsets = WALK_OFFSETS if gait == "walk" else TROT_OFFSETS
        self.swing_frac = swing_fraction or (0.33 if gait == "walk" else 0.45)
        self.control_dt = 1.0 / control_freq
        self.policy_dt = policy_dt
        self.n_control = max(1, round(policy_dt / self.control_dt))
        self.n_sim = max(1, round(self.control_dt / self.model.opt.timestep))
        self.max_steps = int(max_episode_s / (self.n_control * self.control_dt))
        self.target_speed = target_speed
        self.fwd = np.asarray(forward_axis, dtype=float)
        self.fwd /= np.linalg.norm(self.fwd)

        self.w = dict(vel=2.0, lateral=0.5, yaw=0.3, height=1.0, orient=1.0,
                      ang_vel=0.05, energy=0.001, alive=0.2)
        if reward_weights:
            self.w.update(reward_weights)

        # --- indices -----------------------------------------------------
        m = self.model
        jids = m.actuator_trnid[:8, 0]
        self.j_qpos = m.jnt_qposadr[jids]
        self.j_dof = m.jnt_dofadr[jids]

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise RuntimeError("Model needs a free joint on the robot base.")
        self.root_qpos = m.jnt_qposadr[free[0]]
        self.root_dof = m.jnt_dofadr[free[0]]
        self.root_body = m.jnt_bodyid[free[0]]

        # --- spaces ------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (34,), np.float32)

        # default (normal) param vector, in physical units
        self.default_params = np.array([0.687, 0.35, 0.35, -1.0, -1.3, 1.4])
        self.params = self.default_params.copy()
        self.phase = 0.0
        self.step_count = 0
        self.target_height = 0.0
        self.np_random_ = np.random.default_rng(seed)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _to_phys(a):
        return PARAM_LOW + (np.asarray(a) + 1.0) * 0.5 * (PARAM_HIGH - PARAM_LOW)

    @staticmethod
    def _to_norm(p):
        return 2.0 * (p - PARAM_LOW) / (PARAM_HIGH - PARAM_LOW) - 1.0

    def _rot(self):
        return self.data.xmat[self.root_body].reshape(3, 3)   # body -> world

    def _imu(self):
        """Body-frame quantities. If your XML has IMU sensors you can read them from
        data.sensordata instead; free-joint state is equivalent and always available."""
        R = self._rot()
        grav = R.T @ np.array([0.0, 0.0, -1.0])               # (0,0,-1) when level
        lin_w = self.data.qvel[self.root_dof:self.root_dof + 3]
        lin_b = R.T @ lin_w
        gyro = self.data.qvel[self.root_dof + 3:self.root_dof + 6]  # already body frame
        return grav, lin_b, gyro

    def _obs(self):
        d = self.data
        grav, lin_b, gyro = self._imu()
        obs = np.concatenate([
            d.qpos[self.j_qpos],
            0.1 * d.qvel[self.j_dof],
            grav, lin_b, gyro,
            [d.qpos[self.root_qpos + 2]],
            [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
            self._to_norm(self.params),
        ])
        return obs.astype(np.float32)

    def _apply_gait(self):
        p = dict(zip(PARAM_NAMES, self.params))
        for i, leg in enumerate(LEGS):
            sweep = p["sweep_left"] if leg["left"] else p["sweep_right"]
            ph = (self.phase + self.offsets[i]) % 1.0
            sh, kn = compute_leg(p, sweep, ph, self.swing_frac)
            self.data.ctrl[leg["sh"]] = sh
            self.data.ctrl[leg["kn"]] = kn

    def _apply_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # set once, directly: no EMA smoothing, since there is nothing to smooth between
        self.params = self._to_phys(action)
        return action

    def _calculate_reward(self, speeds, lats, yaws, heights, orients, angs, energy):
        vx = np.mean(speeds)
        reward_terms = {
            "vel": self.w["vel"] * np.exp(-((vx - self.target_speed) ** 2) / 0.05),
            "lateral": -self.w["lateral"] * abs(np.mean(lats)),
            "yaw": -self.w["yaw"] * abs(np.mean(yaws)),
            "height": -self.w["height"] * abs(np.mean(heights) - self.target_height)
            / max(self.target_height, 1e-3),
            "orient": -self.w["orient"] * np.mean(orients),
            "ang_vel": -self.w["ang_vel"] * np.mean(angs),
            "energy": -self.w["energy"] * energy / self.n_control,
            "alive": self.w["alive"],
        }
        return float(sum(reward_terms.values())), reward_terms, vx

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        self.params = self.default_params.copy()
        self.phase = 0.0
        self.step_count = 0

        # home pose + settle (as in set_home)
        for leg in LEGS:
            self.data.ctrl[leg["sh"]] = self.params[0]
            self.data.ctrl[leg["kn"]] = self.params[3]
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self.target_height = float(self.data.qpos[self.root_qpos + 2])

        # small random initial phase so the policy sees varied states
        self.phase = float(self.np_random_.uniform(0, 1))
        self._apply_gait()
        for _ in range(self.n_sim):
            mujoco.mj_step(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        """One policy action = one full episode with fixed gait parameters."""
        action = self._apply_action(action)

        pos0 = self.data.qpos[self.root_qpos:self.root_qpos + 3].copy()
        lat_axis = np.array([-self.fwd[1], self.fwd[0], 0.0])

        total_reward = 0.0
        terminated = False
        vx_hist = []
        term_sums = {}

        for _ in range(self.max_steps):          # max_steps = chunks of policy_dt
            energy = 0.0
            speeds, lats, yaws, heights, orients, angs = [], [], [], [], [], []

            for _ in range(self.n_control):
                self.phase = (self.phase + self.control_dt / self.params[5]) % 1.0
                self._apply_gait()               # params stay constant, only phase advances
                for _ in range(self.n_sim):
                    mujoco.mj_step(self.model, self.data)

                grav, lin_b, gyro = self._imu()
                speeds.append(lin_b @ self.fwd)
                lats.append(lin_b @ lat_axis)
                yaws.append(gyro[2])
                angs.append(np.sum(gyro[:2] ** 2))
                heights.append(self.data.qpos[self.root_qpos + 2])
                orients.append(np.sum(grav[:2] ** 2))
                energy += float(np.sum(np.abs(
                    self.data.actuator_force[:8] * self.data.qvel[self.j_dof])))

                if self.render_mode == "human":
                    self.render()

            r_chunk, terms, vx = self._calculate_reward(
                speeds, lats, yaws, heights, orients, angs, energy)
            vx_hist.append(vx)
            for k, v in terms.items():
                term_sums[k] = term_sums.get(k, 0.0) + v

            # ---- termination checks (once per chunk)
            grav, _, _ = self._imu()
            h = self.data.qpos[self.root_qpos + 2]
            obs = self._obs()
            if not (np.all(np.isfinite(obs)) and np.isfinite(r_chunk)):
                # physics blew up: NaN/inf state would otherwise crash the policy update
                terminated, r_chunk = True, 0.0
            elif grav[2] > -0.5 or h < 0.5 * self.target_height:
                terminated = True                # tilted >60 deg / collapsed

            total_reward += r_chunk
            if terminated:
                total_reward -= 5.0
                break

        self.step_count += 1
        obs = np.nan_to_num(self._obs(), nan=0.0, posinf=0.0, neginf=0.0)
        truncated = not terminated               # survived the full episode
        pos1 = self.data.qpos[self.root_qpos:self.root_qpos + 3]
        info = {"reward_terms": term_sums,
                "forward_vel": float(np.mean(vx_hist)),
                "distance": float(np.linalg.norm(pos1[:2] - pos0[:2])),
                "params": self.params.copy()}
        return obs, float(total_reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None