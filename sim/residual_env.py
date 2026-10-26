"""
Closed-loop residual RL environment: Bezier gait + learned corrections, for push recovery.

  Every control step (50 Hz) the fixed Bezier gait (bezier_helpers.compute_leg) gives the
  joint targets for all legs. The policy reads what the REAL robot can sense and adds a small
  correction (delta) to every actuator, including the sideways hip (HAA) joints:

      ctrl = bezier_target + delta_scale * action          action in [-1, 1]^nu

  obs (only sensors the hardware has, so the policy can run on the robot unchanged):
      projected gravity (3)   BNO055 orientation -> which way is down, in the body frame
      gyro (3)                BNO055 angular rates
      gait phase sin/cos (2)  the gait clock
      previous action (nu)    what the policy did last step
  (no joint encoders on hobby servos, no body velocity: those are only used in the reward)

  Robustness comes from training under disturbances, re-drawn every episode:
      random pushes on the trunk, friction, trunk mass, motor strength (kp),
      sensor noise, and 0-1 step action latency.

  The base gait parameters are the 6 RL gait numbers: by default the hand-tuned values; pass
  gait_params=load_gait_params("models/best_params.json") to build on the best ones found by
  `python -m sim.rl_ppo train`.
"""

import json
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .bezier_helpers import PARAM_NAMES, TROT_OFFSETS, WALK_OFFSETS, compute_leg, legs_for_model

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "assets", "scene.xml")
BEST_PARAMS = os.path.join(HERE, "..", "models", "best_params.json")

DEFAULT_GAIT_PARAMS = dict(shoulder_home=0.687, sweep_left=0.35, sweep_right=0.35,
                           knee_home=-1.0, knee_lift=-1.3, cycle_period=1.4)


def load_gait_params(path=BEST_PARAMS):
    """The 6 gait numbers from a best_params.json written by sim.rl_ppo."""
    with open(path) as f:
        params = json.load(f)["params"]
    return {k: float(params[k]) for k in PARAM_NAMES}


class ResidualGaitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path=DEFAULT_XML,
        render_mode=None,
        gait="trot",
        gait_params=None,             # dict of the 6 gait numbers; None -> hand-tuned defaults
        control_freq=50.0,            # Hz, policy + Bezier update rate
        max_episode_s=10.0,
        delta_scale=(0.25, 0.25, 0.25),   # max correction [rad] for (haa, shoulder, knee)
        target_speed=0.15,            # m/s forward (what the hand-tuned trot does)
        # disturbances (set randomize=False / push_force=(0, 0) to switch them off)
        randomize=True,
        push_force=(30.0, 110.0),     # N, horizontal push (robot ~11 kg in sim; plain Bezier falls >~80 N)
        push_duration=0.15,           # s
        push_interval=(1.5, 3.0),     # s between pushes
        reward_weights=None,
        seed=None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        m = self.model

        self.legs = legs_for_model(m)
        self.offsets = WALK_OFFSETS if gait == "walk" else TROT_OFFSETS
        self.swing_frac = 0.33 if gait == "walk" else 0.45
        self.gait_params = dict(gait_params or DEFAULT_GAIT_PARAMS)

        self.control_dt = 1.0 / control_freq
        self.n_sim = max(1, round(self.control_dt / m.opt.timestep))
        self.max_steps = int(max_episode_s / self.control_dt)
        self.target_speed = target_speed

        # every actuator gets a residual; scale depends on the joint type
        self.nu = m.nu
        scale = np.full(m.nu, delta_scale[1])
        for leg in self.legs:
            if leg["haa"] is not None:
                scale[leg["haa"]] = delta_scale[0]
            scale[leg["sh"]] = delta_scale[1]
            scale[leg["kn"]] = delta_scale[2]
        self.delta_scale = scale
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.root_qpos = m.jnt_qposadr[free]
        self.root_dof = m.jnt_dofadr[free]
        self.trunk = m.jnt_bodyid[free]

        # nominal values for domain randomization
        self.nominal_friction = m.geom_friction.copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_gain = m.actuator_gainprm.copy()
        self.nominal_bias = m.actuator_biasprm.copy()

        self.randomize = randomize
        self.push_force = push_force
        self.push_duration = push_duration
        self.push_interval = push_interval

        self.w = dict(vel=1.0, upright=1.0, height=1.0, yaw=0.2, delta=0.05,
                      smooth=0.1, alive=0.5, fall=10.0)
        if reward_weights:
            self.w.update(reward_weights)

        obs_dim = 3 + 3 + 2 + self.nu
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ sensing
    def _gravity_body(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _obs(self):
        grav = self._gravity_body()
        gyro = self.data.qvel[self.root_dof + 3:self.root_dof + 6].copy()
        if self.randomize:
            grav = grav + self.rng.normal(0, 0.02, 3)
            gyro = gyro + self.rng.normal(0, 0.05, 3)
        obs = np.concatenate([grav, gyro,
                              [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
                              self.prev_action])
        return obs.astype(np.float32)

    # ------------------------------------------------------------ gait + disturbances
    def _bezier_targets(self):
        p = self.gait_params
        ctrl = np.zeros(self.nu)
        for i, leg in enumerate(self.legs):
            sweep = p["sweep_left"] if leg["left"] else p["sweep_right"]
            ph = (self.phase + self.offsets[i]) % 1.0
            ctrl[leg["sh"]], ctrl[leg["kn"]] = compute_leg(p, sweep, ph, self.swing_frac)
        return ctrl                       # HAA targets stay 0 (legs straight down)

    def _randomize_model(self):
        m, r = self.model, self.rng
        m.geom_friction[:] = self.nominal_friction
        m.body_mass[:] = self.nominal_mass
        m.body_inertia[:] = self.nominal_inertia
        m.actuator_gainprm[:] = self.nominal_gain
        m.actuator_biasprm[:] = self.nominal_bias
        if not self.randomize:
            return
        m.geom_friction[:, 0] *= r.uniform(0.5, 1.25)   # MuJoCo uses the max of both surfaces
        k = r.uniform(0.8, 1.25)                     # trunk mass / payload
        m.body_mass[self.trunk] *= k
        m.body_inertia[self.trunk] *= k
        s = r.uniform(0.8, 1.2)                      # motor strength: kp of the position servos
        m.actuator_gainprm[:, 0] *= s
        m.actuator_biasprm[:, 1] *= s

    def _schedule_push(self):
        lo, hi = self.push_interval
        self.next_push = self.data.time + self.rng.uniform(lo, hi)

    def _apply_push(self):
        d = self.data
        d.xfrc_applied[self.trunk, :] = 0.0
        if self.push_force[1] <= 0:
            return
        if self.push_end is not None and d.time < self.push_end:
            d.xfrc_applied[self.trunk, :3] = self.push_vec
            return
        self.push_end = None
        if d.time >= self.next_push:
            a = self.rng.uniform(0, 2 * np.pi)
            f = self.rng.uniform(*self.push_force)
            self.push_vec = np.array([f * np.cos(a), f * np.sin(a), 0.0])
            self.push_end = d.time + self.push_duration
            d.xfrc_applied[self.trunk, :3] = self.push_vec
            self.n_pushes += 1
            self._schedule_push()

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_model()
        mujoco.mj_resetData(self.model, self.data)

        self.phase = float(self.rng.uniform(0, 1)) if self.randomize else 0.0
        self.prev_action = np.zeros(self.nu)
        self.pending_action = np.zeros(self.nu)
        self.latency = int(self.rng.integers(0, 2)) if self.randomize else 0
        self.step_count = 0
        self.push_end = None
        self.n_pushes = 0

        # stand, settle, then start the gait
        stand = np.zeros(self.nu)
        for leg in self.legs:
            stand[leg["sh"]] = self.gait_params["shoulder_home"]
            stand[leg["kn"]] = self.gait_params["knee_home"]
        self.data.ctrl[:] = stand
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.target_height = float(self.data.qpos[self.root_qpos + 2])
        self._schedule_push()
        self.next_push = max(self.next_push, self.data.time + 1.0)   # first second: no push
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # servo latency: optionally apply the previous step's action
        applied = self.pending_action if self.latency else action
        self.pending_action = action

        self.phase = (self.phase + self.control_dt / self.gait_params["cycle_period"]) % 1.0
        ctrl = self._bezier_targets() + self.delta_scale * applied
        self.data.ctrl[:] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        for _ in range(self.n_sim):
            self._apply_push()
            mujoco.mj_step(self.model, self.data)

        # ------------------------------------------------ reward (privileged sim state allowed)
        R = self.data.xmat[self.trunk].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        v_body = R.T @ self.data.qvel[self.root_dof:self.root_dof + 3]
        yaw_rate = self.data.qvel[self.root_dof + 5]
        h = self.data.qpos[self.root_qpos + 2]
        terms = {
            "vel": self.w["vel"] * np.exp(-((v_body[0] - self.target_speed) ** 2) / 0.01),
            "upright": -self.w["upright"] * float(np.sum(grav[:2] ** 2)),
            "height": -self.w["height"] * abs(h - self.target_height) / self.target_height,
            "yaw": -self.w["yaw"] * yaw_rate ** 2,
            "delta": -self.w["delta"] * float(np.mean(action ** 2)),     # stay close to Bezier
            "smooth": -self.w["smooth"] * float(np.mean((action - self.prev_action) ** 2)),
            "alive": self.w["alive"],
        }
        reward = float(sum(terms.values()))
        self.prev_action = action
        self.step_count += 1

        terminated = bool(grav[2] > -0.5 or h < 0.5 * self.target_height
                          or not np.all(np.isfinite(self.data.qpos)))
        if terminated:
            reward -= self.w["fall"]
        truncated = self.step_count >= self.max_steps
        obs = np.nan_to_num(self._obs())
        info = {"reward_terms": terms, "fwd_vel": float(v_body[0]), "pushes": self.n_pushes}

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

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
