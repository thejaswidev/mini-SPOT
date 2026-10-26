"""
Closed-loop RL that slightly reshapes the Bezier curve, for uneven terrain, slopes and pushes.

  Idea from D^2-GMBC (Rahme et al. 2020, arXiv 2010.12070): keep the Bezier gait, but let a
  policy change the curve's own parameters a little every control step, and train it on random
  terrain + dynamics. An action of 0 is exactly the plain Bezier gait.

  Per leg, 5 small changes to the curve (all bounded, see MOD_LIMITS):
      clearance   swing lift height x (1 + c)                 paper: psi   (clearance height)
      depth       stance knee dip: foot pushed slightly into  paper: delta (virtual ground
                  the ground (+) or held above it (-)                       penetration depth)
      shoulder    shift of the whole loop forward / back       paper: foot residual, x
      knee        leg a bit longer / shorter                   paper: foot residual, z
      haa         sideways hip angle                           paper: foot residual, y
  (on the 8-actuator model without HAA there are 4 per leg)

  obs (only the BNO055 IMU + the gait clock, so the policy can run on the robot unchanged):
      gravity direction in the body frame (3), gyro (3), gait phase sin/cos (2), last action

  Training conditions, re-drawn every episode:
      terrain     random bumps 0-6 cm (assets/scene_terrain.xml heightfield)
      slopes      up/downhill +-15 deg, sideways +-8 deg (gravity is tilted: same physics)
      pushes      off by default (push_force=(30, 100) to add them)
      dynamics    friction, trunk mass, motor strength (kp), sensor noise, 0-1 step latency

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

from .bezier_helpers import (KNEE_MAX, KNEE_MIN, PARAM_NAMES, TROT_OFFSETS, WALK_OFFSETS,
                             cubic_bezier, legs_for_model, lerp)

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT_XML = os.path.join(HERE, "..", "assets", "scene.xml")
TERRAIN_XML = os.path.join(HERE, "..", "assets", "scene_terrain.xml")
BEST_PARAMS = os.path.join(HERE, "..", "models", "best_params.json")

DEFAULT_GAIT_PARAMS = dict(shoulder_home=0.687, sweep_left=0.35, sweep_right=0.35,
                           knee_home=-1.0, knee_lift=-1.3, cycle_period=1.4)

# max change per curve parameter for action = +-1 (kept small on purpose)
MOD_LIMITS = dict(
    clearance=0.5,    # fraction of the normal knee lift (0.3 rad) -> +-0.15 rad
    depth=0.05,       # rad, knee dip during stance
    shoulder=0.1,     # rad, shift of the swing/stance loop
    knee=0.1,         # rad, knee home offset
    haa=0.1,          # rad, sideways hip
)


def load_gait_params(path=BEST_PARAMS):
    """The 6 gait numbers from a best_params.json written by sim.rl_ppo."""
    with open(path) as f:
        params = json.load(f)["params"]
    return {k: float(params[k]) for k in PARAM_NAMES}


def modulated_leg(p, sweep, phase, swing_frac, clearance=0.0, depth=0.0, shoulder=0.0, knee=0.0):
    """bezier_helpers.compute_leg with the curve reshaped slightly (all changes 0 = original).

    clearance scales the knee lift of the swing Bezier; depth adds a sine dip to the knee
    during stance; shoulder / knee shift the whole loop.
    """
    home = p["shoulder_home"] + shoulder
    kh = p["knee_home"] + knee
    kl = kh + (p["knee_lift"] - p["knee_home"]) * (1.0 + clearance)
    fwd, back = home - sweep, home + sweep
    if phase < swing_frac:                               # swing: Bezier curve
        t = phase / swing_frac
        sh = cubic_bezier(back, back, fwd, fwd, t)
        kn = cubic_bezier(kh, kl, kl, kh, t)
    else:                                                # stance: push back (+ depth dip)
        s = (phase - swing_frac) / (1.0 - swing_frac)
        sh = lerp(fwd, back, s)
        kn = kh + depth * np.sin(np.pi * s)
    return np.clip(sh, 0.0, np.pi), np.clip(kn, KNEE_MIN, KNEE_MAX)


def _smooth_noise(rng, n, cells):
    """n x n smooth random field in [0, 1]: random grid of `cells` points, bilinear upsampled."""
    coarse = rng.uniform(0.0, 1.0, (cells, cells))
    x = np.linspace(0, cells - 1, n)
    rows = np.array([np.interp(x, np.arange(cells), r) for r in coarse])           # (cells, n)
    return np.array([np.interp(x, np.arange(cells), c) for c in rows.T]).T        # (n, n)


class ResidualGaitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        gait="trot",
        gait_params=None,             # dict of the 6 gait numbers; None -> hand-tuned defaults
        control_freq=50.0,            # Hz, policy + Bezier update rate
        max_episode_s=10.0,
        target_speed=0.15,            # m/s forward (what the hand-tuned trot does on flat ground)
        # training conditions; each is a (low, high) range drawn per episode. (0, 0) = off
        bump_height=(0.0, 0.06),      # m, terrain bumps (needs the terrain scene)
        slope_deg=(-15.0, 15.0),      # uphill (+) / downhill (-) along the walking direction
        side_slope_deg=(-8.0, 8.0),
        push_force=(0.0, 0.0),        # N, horizontal push; off by default, e.g. (30, 100) to train with pushes
        push_duration=0.15,           # s
        push_interval=(1.5, 3.0),     # s between pushes
        randomize=True,               # friction / mass / motor strength / noise / latency
        terrain=True,                 # False: flat floor scene (faster, no bumps)
        reward_weights=None,
        seed=None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(TERRAIN_XML if terrain else FLAT_XML)
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

        # action layout: per leg [clearance, depth, shoulder, knee(, haa)]
        self.mod_names = ["clearance", "depth", "shoulder", "knee"]
        self.has_haa = all(leg["haa"] is not None for leg in self.legs)
        if self.has_haa:
            self.mod_names.append("haa")
        self.n_mod = len(self.mod_names)
        self.act_dim = self.n_mod * len(self.legs)
        self.mod_scale = np.array([MOD_LIMITS[n] for n in self.mod_names])
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.root_qpos = m.jnt_qposadr[free]
        self.root_dof = m.jnt_dofadr[free]
        self.trunk = m.jnt_bodyid[free]

        self.hfield = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if self.hfield >= 0:
            self.terrain_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
            self.hf_nrow, self.hf_ncol = m.hfield_nrow[self.hfield], m.hfield_ncol[self.hfield]
            self.hf_size = m.hfield_size[self.hfield].copy()     # radius x, radius y, height, base

        # nominal values for domain randomization
        self.nominal_friction = m.geom_friction.copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_gain = m.actuator_gainprm.copy()
        self.nominal_bias = m.actuator_biasprm.copy()

        self.randomize = randomize
        self.bump_height = bump_height if self.hfield >= 0 else (0.0, 0.0)
        self.slope_deg = slope_deg
        self.side_slope_deg = side_slope_deg
        self.push_force = push_force
        self.push_duration = push_duration
        self.push_interval = push_interval

        self.w = dict(vel=1.0, upright=1.0, height=1.0, yaw=0.2, mod=0.05,
                      smooth=0.1, alive=0.5, fall=10.0)
        if reward_weights:
            self.w.update(reward_weights)

        obs_dim = 3 + 3 + 2 + self.act_dim
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ terrain
    def _make_terrain(self):
        if self.hfield < 0:
            return
        amp = self.rng.uniform(*self.bump_height)
        n = self.hf_nrow
        cells = int(self.rng.integers(20, 50))                 # bump size ~16-40 cm
        field = _smooth_noise(self.rng, n, cells) * amp / self.hf_size[2]
        self.model.hfield_data[:] = np.clip(field, 0.0, 1.0).ravel()
        # put the ground under the start position at z = 0
        self.model.geom_pos[self.terrain_geom, 2] = 0.0
        self.model.geom_pos[self.terrain_geom, 2] = -self._ground_height(0.0, 0.0)
        if self.viewer is not None:
            self.viewer.update_hfield(self.hfield)

    def _ground_height(self, x, y):
        """Terrain height below (x, y) in world coordinates (0 on the flat scene)."""
        if self.hfield < 0:
            return 0.0
        rx, ry, h = self.hf_size[:3]
        c = int(round((x + rx) / (2 * rx) * (self.hf_ncol - 1)))
        r = int(round((y + ry) / (2 * ry) * (self.hf_nrow - 1)))
        c, r = min(max(c, 0), self.hf_ncol - 1), min(max(r, 0), self.hf_nrow - 1)
        return float(self.model.geom_pos[self.terrain_geom, 2]
                     + self.model.hfield_data[r * self.hf_ncol + c] * h)

    def _set_slope(self):
        """Tilt gravity: + pitch = uphill when walking forward (+x)."""
        th = np.radians(self.rng.uniform(*self.slope_deg))
        ph = np.radians(self.rng.uniform(*self.side_slope_deg))
        self.model.opt.gravity[:] = 9.81 * np.array(
            [-np.sin(th), np.cos(th) * np.sin(ph), -np.cos(th) * np.cos(ph)])
        self.slope = (float(np.degrees(th)), float(np.degrees(ph)))

    # ------------------------------------------------------------ sensing
    def _obs(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        g = self.model.opt.gravity
        grav = R.T @ (g / np.linalg.norm(g))       # what the IMU feels as "down"
        gyro = self.data.qvel[self.root_dof + 3:self.root_dof + 6].copy()
        if self.randomize:
            grav = grav + self.rng.normal(0, 0.02, 3)
            gyro = gyro + self.rng.normal(0, 0.05, 3)
        obs = np.concatenate([grav, gyro,
                              [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
                              self.prev_action])
        return obs.astype(np.float32)

    def _body_height(self):
        x, y, z = self.data.qpos[self.root_qpos:self.root_qpos + 3]
        return float(z - self._ground_height(x, y))

    # ------------------------------------------------------------ gait + disturbances
    def _gait_targets(self, action):
        """Bezier targets with the curve of each leg reshaped by its slice of the action."""
        p = self.gait_params
        ctrl = np.zeros(self.model.nu)
        mods = action.reshape(len(self.legs), self.n_mod) * self.mod_scale
        for i, leg in enumerate(self.legs):
            mod = dict(zip(self.mod_names, mods[i]))
            sweep = p["sweep_left"] if leg["left"] else p["sweep_right"]
            ph = (self.phase + self.offsets[i]) % 1.0
            ctrl[leg["sh"]], ctrl[leg["kn"]] = modulated_leg(
                p, sweep, ph, self.swing_frac, mod["clearance"], mod["depth"],
                mod["shoulder"], mod["knee"])
            if leg["haa"] is not None:
                ctrl[leg["haa"]] = mod.get("haa", 0.0)
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

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
        self._make_terrain()
        self._set_slope()
        mujoco.mj_resetData(self.model, self.data)

        self.phase = float(self.rng.uniform(0, 1)) if self.randomize else 0.0
        self.prev_action = np.zeros(self.act_dim)
        self.pending_action = np.zeros(self.act_dim)
        self.latency = int(self.rng.integers(0, 2)) if self.randomize else 0
        self.step_count = 0
        self.push_end = None
        self.n_pushes = 0

        # stand, settle, then start the gait
        stand = np.zeros(self.model.nu)
        for leg in self.legs:
            stand[leg["sh"]] = self.gait_params["shoulder_home"]
            stand[leg["kn"]] = self.gait_params["knee_home"]
        self.data.ctrl[:] = stand
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.target_height = self._body_height()
        self._schedule_push()
        self.next_push = max(self.next_push, self.data.time + 1.0)   # first second: no push
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # servo latency: optionally apply the previous step's action
        applied = self.pending_action if self.latency else action
        self.pending_action = action

        self.phase = (self.phase + self.control_dt / self.gait_params["cycle_period"]) % 1.0
        self.data.ctrl[:] = self._gait_targets(applied)
        for _ in range(self.n_sim):
            self._apply_push()
            mujoco.mj_step(self.model, self.data)

        # ------------------------------------------------ reward (privileged sim state allowed)
        R = self.data.xmat[self.trunk].reshape(3, 3)
        up_body = R.T @ np.array([0.0, 0.0, 1.0])     # ground normal (world z) in the body frame
        v_body = R.T @ self.data.qvel[self.root_dof:self.root_dof + 3]
        yaw_rate = self.data.qvel[self.root_dof + 5]
        h = self._body_height()
        terms = {
            "vel": self.w["vel"] * np.exp(-((v_body[0] - self.target_speed) ** 2) / 0.01),
            "upright": -self.w["upright"] * float(np.sum(up_body[:2] ** 2)),
            "height": -self.w["height"] * abs(h - self.target_height) / self.target_height,
            "yaw": -self.w["yaw"] * yaw_rate ** 2,
            "mod": -self.w["mod"] * float(np.mean(action ** 2)),          # keep changes small
            "smooth": -self.w["smooth"] * float(np.mean((action - self.prev_action) ** 2)),
            "alive": self.w["alive"],
        }
        reward = float(sum(terms.values()))
        self.prev_action = action
        self.step_count += 1

        terminated = bool(up_body[2] < 0.5 or h < 0.5 * self.target_height
                          or not np.all(np.isfinite(self.data.qpos)))
        if terminated:
            reward -= self.w["fall"]
        truncated = self.step_count >= self.max_steps
        obs = np.nan_to_num(self._obs())
        info = {"reward_terms": terms, "fwd_vel": float(v_body[0]), "pushes": self.n_pushes,
                "slope": self.slope}

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
