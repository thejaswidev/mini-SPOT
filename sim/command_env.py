"""
One RL policy for every movement command, following the D^2-GMBC architecture
(Rahme et al. 2020, arXiv 2010.12070, Fig. 2 / Algorithm 1):

    command (vx, vy, wz) ---------------------------+
                                                    v
    IMU + gait clock + command --> policy --> beta = (psi, delta)  --> 3D Bezier gait (sim/bezier3d.py)
                                         \--> delta_f (12)  ----------------(+)--> IK --> 12 servos

  Commands: forward, backward, strafe left / right, turn left / right, stand. They are sampled
  every episode and switched every few seconds, so one network learns all of them and the
  transitions between them.

  action (14, all in [-1, 1]; 0 = the plain paper gait):
      psi    swing clearance height   0.04 m +- 0.02 m
      delta  stance penetration depth 0.005 m +- 0.005 m
      delta_f: x, y, z foot offset per leg, +- 1.5 cm each (kept small on purpose)
  obs: gravity direction (3) + gyro (3) + gait phase sin/cos (2) + command (3) + last action (14)
       (IMU only, like the paper; no foot contact or joint sensors)

  Terrain bumps, slopes, dynamics randomization and optional pushes are inherited from
  ResidualGaitEnv (sim/residual_env.py).
"""

import numpy as np
from gymnasium import spaces
import mujoco

from .bezier3d import DELTA_DEFAULT, PSI_DEFAULT, Bezier3DGait
from .residual_env import ResidualGaitEnv

# command name -> (vx, vy, wz) ranges, body frame [m/s, m/s, rad/s]
COMMANDS = {
    "forward":      ((0.05, 0.20), (0.0, 0.0), (0.0, 0.0)),
    "backward":     ((-0.18, -0.05), (0.0, 0.0), (0.0, 0.0)),
    "strafe_left":  ((0.0, 0.0), (0.04, 0.12), (0.0, 0.0)),
    "strafe_right": ((0.0, 0.0), (-0.12, -0.04), (0.0, 0.0)),
    "turn_left":    ((0.0, 0.0), (0.0, 0.0), (0.2, 0.6)),
    "turn_right":   ((0.0, 0.0), (0.0, 0.0), (-0.6, -0.2)),
    "stand":        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
}
CMD_SCALE = np.array([0.2, 0.12, 0.6])       # normalises the command in the observation

PSI_SCALE, DELTA_SCALE = 0.02, 0.005         # action +-1 -> curve parameter change [m]
RESIDUAL_LIMIT = 0.015                       # m, max foot offset per axis


class CommandGaitEnv(ResidualGaitEnv):
    def __init__(
        self,
        commands=tuple(COMMANDS),     # which commands to train / test on
        command=None,                 # fix one (vx, vy, wz) command instead of sampling
        command_switch=(3.0, 6.0),    # s between command changes within an episode
        t_swing=0.2,                  # s, paper value
        **kwargs,                     # terrain / slopes / pushes / randomize ... (ResidualGaitEnv)
    ):
        super().__init__(**kwargs)
        self.gen = Bezier3DGait(self.model, t_swing=t_swing)
        self.commands = list(commands)
        self.fixed_command = None if command is None else np.asarray(command, dtype=float)
        self.command_switch = command_switch
        self.cmd = np.zeros(3)
        self.cmd_name = "stand"

        self.act_dim = 2 + 12
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3 + 3 + 2 + 3 + self.act_dim,), np.float32)
        self.w.update(lin=1.0, yaw=0.5)

    # ------------------------------------------------------------ commands
    def set_command(self, cmd, name="custom"):
        """Change the command on the fly (e.g. from the keyboard)."""
        self.cmd = np.asarray(cmd, dtype=float)
        self.cmd_name = name

    def _sample_command(self):
        if self.fixed_command is not None:
            self.set_command(self.fixed_command, "fixed")
        else:
            name = self.commands[self.rng.integers(len(self.commands))]
            self.set_command([self.rng.uniform(*r) for r in COMMANDS[name]], name)
        self.next_switch = self.data.time + self.rng.uniform(*self.command_switch)

    # ------------------------------------------------------------ gym API
    def _obs(self):
        base = super()._obs()                     # gravity, gyro, phase, last action
        return np.concatenate([base[:8], self.cmd / CMD_SCALE, base[8:]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.cmd = np.zeros(3)
        super().reset(seed=seed, options=options)
        self.data.ctrl[:] = self.gen.stand_ctrl()
        self._sample_command()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied = self.pending_action if self.latency else action     # servo latency
        self.pending_action = action

        if self.fixed_command is None and self.data.time >= self.next_switch:
            self._sample_command()
        self.phase = (self.phase + self.control_dt / self.gen.t_stride) % 1.0
        psi = PSI_DEFAULT + PSI_SCALE * applied[0]
        delta = DELTA_DEFAULT + DELTA_SCALE * applied[1]
        residual = applied[2:].reshape(4, 3) * RESIDUAL_LIMIT
        feet = self.gen.foot_targets(self.phase, self.cmd, psi, delta, residual)
        self.data.ctrl[:] = self.gen.joint_targets(feet)
        for _ in range(self.n_sim):
            self._apply_push()
            mujoco.mj_step(self.model, self.data)

        # ------------------------------------------------ reward: follow the command
        R = self.data.xmat[self.trunk].reshape(3, 3)
        up_body = R.T @ np.array([0.0, 0.0, 1.0])
        v_body = R.T @ self.data.qvel[self.root_dof:self.root_dof + 3]
        w_body = self.data.qvel[self.root_dof + 3:self.root_dof + 6]
        h = self._body_height()
        terms = {
            "lin": self.w["lin"] * np.exp(-float(np.sum((v_body[:2] - self.cmd[:2]) ** 2)) / 0.005),
            "yaw": self.w["yaw"] * np.exp(-float((w_body[2] - self.cmd[2]) ** 2) / 0.05),
            "upright": -self.w["upright"] * float(np.sum(up_body[:2] ** 2)),
            "height": -self.w["height"] * abs(h - self.target_height) / self.target_height,
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
        info = {"reward_terms": terms, "command": self.cmd_name, "cmd": self.cmd.copy(),
                "vel": np.array([v_body[0], v_body[1], w_body[2]]), "slope": self.slope}
        if self.render_mode == "human":
            self.render()
        return np.nan_to_num(self._obs()), reward, terminated, truncated, info
