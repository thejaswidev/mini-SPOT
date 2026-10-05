"""
Gymnasium environment: RL tunes Bezier gait parameters for a MuJoCo quadruped.

  action  (normalised [-1, 1]^6) -> gait parameters
      0 shoulder_home      rad
      1 sweep_left         rad   (left-side legs  FL, RL)
      2 sweep_right        rad   (right-side legs FR, RR)   -> L/R difference = steering
      3 knee_home          rad
      4 knee_lift          rad
      5 cycle_period       s     (the swing fraction is kept fixed; see swing_fraction arg)

  The Bezier generator runs inside the env at CONTROL_FREQUENCY; the policy
  steps once per `policy_dt` and only changes the *parameters* of the generator.
  Params are low-pass filtered so the gait changes smoothly.

  obs (34): joint pos(8) | joint vel(8) | projected gravity(3) | body lin vel(3)
            | gyro(3) | height(1) | phase sin/cos(2) | current params, normalised(6)
"""

import os
import faulthandler

# Print a Python traceback if we hit a native crash (segfault) instead of dying silently.
faulthandler.enable()

# torch (pulled in by stable-baselines3) and MuJoCo both ship native libs that use
# OpenMP. With multiple OpenMP runtimes / thread pools in one process the Linux
# container can segfault right after "Using cpu device". One thread is plenty for
# this tiny MLP and the sim is single-threaded anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "assets", "scene.xml")

# ---------------------------------------------------------------- gait maths
def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * t

def cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


WALK_OFFSETS = np.array([0.25, 0.75, 0.50, 0.00])   # FR, FL, RR, RL
TROT_OFFSETS = np.array([0.00, 0.50, 0.50, 0.00])

# (low, high) for each action dimension
PARAM_NAMES = ["shoulder_home", "sweep_left", "sweep_right",
               "knee_home", "knee_lift", "cycle_period"]
PARAM_LOW   = np.array([0.20, 0.00, 0.00, -2.69653, -2.69653, 0.50])
PARAM_HIGH  = np.array([1.20, 0.60, 0.60, -0.916298, -0.916298, 2.00])

KNEE_MIN, KNEE_MAX = -2.69653, -0.916298

# ctrl layout: [FR_sh, FR_kn, FL_sh, FL_kn, RR_sh, RR_kn, RL_sh, RL_kn]
LEGS = [
    {"name": "FR", "sh": 0, "kn": 1, "left": False},
    {"name": "FL", "sh": 2, "kn": 3, "left": True},
    {"name": "RR", "sh": 4, "kn": 5, "left": False},
    {"name": "RL", "sh": 6, "kn": 7, "left": True},
]


def compute_leg(p, sweep, phase, swing_frac):
    """Same as the original compute_leg; all legs use inv_sh=True (fwd = home - sweep)."""
    sh_fwd  = p["shoulder_home"] - sweep
    sh_back = p["shoulder_home"] + sweep
    kh, kl = p["knee_home"], p["knee_lift"]

    if phase < swing_frac:
        t = phase / swing_frac
        sh = cubic_bezier(sh_back, sh_back, sh_fwd, sh_fwd, t)
        kn = cubic_bezier(kh, kl, kl, kh, t)
    else:
        s = (phase - swing_frac) / (1.0 - swing_frac)
        sh = lerp(sh_fwd, sh_back, s)
        kn = kh
    return np.clip(sh, 0.0, np.pi), np.clip(kn, KNEE_MIN, KNEE_MAX)


# ---------------------------------------------------------------- environment
class QuadrupedBezierEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path=DEFAULT_XML,        # resolved relative to this file, not the cwd
        render_mode=None,
        gait="walk",                 # "walk" | "trot"
        swing_fraction=None,         # default 0.33 walk / 0.45 trot
        control_freq=50.0,           # Hz, Bezier target update rate
        policy_dt=0.2,               # s between policy actions
        max_episode_s=20.0,
        target_speed=0.3,            # m/s along body forward axis
        forward_axis=(1.0, 0.0, 0.0),  # body-frame forward direction  (CHECK for your model!)
        param_smoothing=0.3,         # EMA factor on params (1 = no smoothing)
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
        self.alpha = param_smoothing

        self.w = dict(vel=2.0, lateral=0.5, yaw=0.3, height=1.0, orient=1.0,
                      ang_vel=0.05, energy=0.001, action_rate=0.05, alive=0.2)
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
        self.prev_action = np.zeros(6)
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

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        self.params = self.default_params.copy()
        self.phase = 0.0
        self.prev_action[:] = 0.0
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
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # smooth parameter update (EMA toward requested params)
        target = self._to_phys(action)
        self.params = (1 - self.alpha) * self.params + self.alpha * target

        pos0 = self.data.qpos[self.root_qpos:self.root_qpos + 3].copy()
        energy = 0.0
        speeds, lats, yaws, heights, orients, angs = [], [], [], [], [], []

        for _ in range(self.n_control):
            self.phase = (self.phase + self.control_dt / self.params[5]) % 1.0
            self._apply_gait()
            for _ in range(self.n_sim):
                mujoco.mj_step(self.model, self.data)

            grav, lin_b, gyro = self._imu()
            speeds.append(lin_b @ self.fwd)
            # lateral = body-frame horizontal axis perpendicular to forward
            lat_axis = np.array([-self.fwd[1], self.fwd[0], 0.0])
            lats.append(lin_b @ lat_axis)
            yaws.append(gyro[2])
            angs.append(np.sum(gyro[:2] ** 2))
            heights.append(self.data.qpos[self.root_qpos + 2])
            orients.append(np.sum(grav[:2] ** 2))
            energy += float(np.sum(np.abs(self.data.actuator_force[:8] * self.data.qvel[self.j_dof])))

        # ------------------------------------------------------------ reward
        vx = np.mean(speeds)
        r = {
            "vel":     self.w["vel"] * np.exp(-((vx - self.target_speed) ** 2) / 0.05),
            "lateral": -self.w["lateral"] * abs(np.mean(lats)),
            "yaw":     -self.w["yaw"] * abs(np.mean(yaws)),
            "height":  -self.w["height"] * abs(np.mean(heights) - self.target_height) / max(self.target_height, 1e-3),
            "orient":  -self.w["orient"] * np.mean(orients),
            "ang_vel": -self.w["ang_vel"] * np.mean(angs),
            "energy":  -self.w["energy"] * energy / self.n_control,
            "action_rate": -self.w["action_rate"] * np.sum((action - self.prev_action) ** 2),
            "alive":   self.w["alive"],
        }
        reward = float(sum(r.values()))
        self.prev_action = action

        # ------------------------------------------------------------ done
        grav, _, _ = self._imu()
        h = self.data.qpos[self.root_qpos + 2]
        terminated = bool(grav[2] > -0.5 or h < 0.5 * self.target_height)  # tilted >60 deg / collapsed
        # physics blew up: NaN/inf state would otherwise crash the policy update
        obs = self._obs()
        if not (np.all(np.isfinite(obs)) and np.isfinite(reward)):
            terminated = True
            reward = 0.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if terminated:
            reward -= 5.0
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        pos1 = self.data.qpos[self.root_qpos:self.root_qpos + 3]
        info = {"reward_terms": r, "forward_vel": vx,
                "distance": float(np.linalg.norm(pos1[:2] - pos0[:2])), "params": self.params.copy()}

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


# ---------------------------------------------------------------- quick test / training
if __name__ == "__main__":
    import sys
    training = len(sys.argv) > 1 and sys.argv[1] == "train"
    if training:
        # Initialise torch before any MuJoCo model is created, and pin it to one
        # thread (see OMP_NUM_THREADS note at the top of the file).
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        print("import done")

    env = QuadrupedBezierEnv()
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape)

    if training:
        venv = make_vec_env(lambda: QuadrupedBezierEnv(), n_envs=1)
        model = PPO("MlpPolicy", venv, verbose=1, n_steps=256, batch_size=256,
                    learning_rate=3e-4, gamma=0.97, ent_coef=0.005)
        print("env initialised")
        model.learn(total_timesteps=500_000)
        model.save("quadruped_bezier_ppo")
    else:
        # sanity: zero action = hold current (default) gait params
        total = 0.0
        for _ in range(50):
            obs, rew, term, trunc, info = env.step(env.action_space.sample() * 0.0)
            total += rew
            if term or trunc:
                break
        print("return:", total, "fwd vel:", info["forward_vel"])