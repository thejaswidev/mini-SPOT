"""Bezier gait parameters and helper functions."""

import numpy as np


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def cubic_bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


WALK_OFFSETS = np.array([0.25, 0.75, 0.50, 0.00])  # FR, FL, RR, RL
TROT_OFFSETS = np.array([0.00, 0.50, 0.50, 0.00])

# (low, high) for each action dimension
PARAM_NAMES = [
    "shoulder_home",
    "sweep_left",
    "sweep_right",
    "knee_home",
    "knee_lift",
    "cycle_period",
]
PARAM_LOW = np.array([0.20, 0.00, 0.00, -2.69653, -2.69653, 0.50])
PARAM_HIGH = np.array([1.20, 0.60, 0.60, -0.916298, -0.916298, 2.00])

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
    sh_fwd = p["shoulder_home"] - sweep
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
