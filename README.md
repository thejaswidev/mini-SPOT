# Mini-SPOT

A small 4-legged robot dog (inspired by Boston Dynamics Spot), built as a
Bachelor's project in Robotics Engineering.

Right now everything runs in a **physics simulation** (MuJoCo) on the computer.
The robot can walk, trot, and turn left/right on command. Reinforcement
learning (RL) is being added to find a smoother, better walking style.

> This README describes the `rl-gait` branch.

---

## The 30-second version

- The robot has **4 legs**, each with **2 motors**: a **shoulder** (swings the leg forward/back)
  and a **knee** (bends the leg). 4 × 2 = **8 motors**.
- Walking = moving each leg in a repeating loop: **lift it, swing it forward, put it down, push back**.
  Doing this with the 4 legs at different times makes the body move forward.
- **Turning** = left legs and right legs push in opposite directions, so the body spins on the spot.
- **RL** = the computer tries thousands of different walking settings (step length, speed, knee bend…)
  and keeps the ones that make the robot walk fast, straight and steady.

---

## What each file does

### The files you actually run

| File | What it is | When to use it |
|---|---|---|
| `sim/gait_controller.py` | **Drive the robot yourself.** Opens a 3D window; you type `walk`, `turn left`, `stop`… | To watch the robot move and test gaits by hand |
| `sim/rl_ppo.py` | **Train / test the RL brain.** Lets the computer learn the best walking settings | To improve the walk automatically |
| `sim/mujoco_env.py` | **Just show the robot standing.** The simplest viewer | To check the robot model loads |
| `tests/test_gait_controller.py` | **Automatic checks.** Turns the robot 45°/90° without a window and checks it really turned that much | After changing code, to make sure nothing broke |

### The files the others use (you don't run these directly)

| File | What it is |
|---|---|
| `sim/bezier_helpers.py` | **The gait math.** Says where every leg should be at every moment (the smooth leg curves and the timing between legs). Shared by both the manual controller and RL. |
| `sim/gym_class.py` | **The RL training ground.** Takes a set of walking settings, lets the robot walk with them for 20 s in the simulation, and gives back a **score**. |
| `assets/mini_spot.xml` | **The robot itself**: body, legs, motors, sensors, joint limits. |
| `assets/scene.xml` | **The world**: floor, light, sky. It loads the robot. |
| `assets/CAD_Files/` | 3D shapes of the real robot parts (used to draw the robot). |

### Setup files

| File | What it is |
|---|---|
| `docker/Dockerfile` | Recipe for a ready-made Linux box with Python, MuJoCo, PyTorch and RL libraries installed |
| `docker/docker-compose.yml` | Settings for that box (shares this folder and your screen with it) |
| `run.sh` | One command that starts the box and gives you a terminal inside it |

---

## How the files are connected

```
                      assets/scene.xml  ──loads──►  assets/mini_spot.xml
                      (the world)                   (the robot)
                             ▲
                             │ every script loads the world + robot
        ┌────────────────────┼─────────────────────────┐
        │                    │                         │
 sim/mujoco_env.py   sim/gait_controller.py      sim/gym_class.py ◄── used by ── sim/rl_ppo.py
 (just stands)       (YOU give commands)         (scores a walk)                 (RL learns)
                             │                         │
                             └──────── both use ───────┘
                                          ▼
                                sim/bezier_helpers.py
                                (where each leg goes, when)
```

**Two separate ways to move the robot:**

1. **Manual** – `gait_controller.py`: fixed rules. You give commands, it walks / turns.
2. **Learned** – `rl_ppo.py` + `gym_class.py`: RL searches for the best walking settings.

Turning is **only manual** on purpose. RL only learns forward walking, so turning
can't mess up the walking training. The plan: let RL find the best walking settings,
then put them into the manual controller (`set …` command), and keep turning as is.

---

## How to run it

### On the Linux PC (Docker – recommended)

First time, or after the `Dockerfile` changes:
```bash
git clone -b rl-gait https://github.com/thejaswidev/mini-SPOT
cd mini-SPOT
docker compose -f docker/docker-compose.yml build
```

Every time:
```bash
./run.sh                    # you are now inside the box
```

Then, inside the box:
```bash
python sim/gait_controller.py                            # drive the robot
python -m sim.rl_ppo train                               # train RL
python -m sim.rl_ppo test --model quadruped_bezier_ppo   # watch the trained result
python -m pytest tests                                   # run the checks
```

### On a Mac (without Docker)

One-time setup:
```bash
python3 -m venv .venv
.venv/bin/pip install mujoco==3.1.6 "numpy<2" gymnasium stable-baselines3 torch pytest
```

Then use `.venv/bin/mjpython` for anything that opens a 3D window (Mac needs it),
and `.venv/bin/python` for everything else:
```bash
.venv/bin/mjpython sim/gait_controller.py
.venv/bin/python -m sim.rl_ppo train
.venv/bin/python -m pytest tests
```

> Run commands from the `mini-SPOT` folder, in a normal terminal (you need to type into it).

---

## Using the controller (`gait_controller.py`)

A 3D window opens with the robot standing. **Type commands in the terminal** and press Enter:

| Command | What happens |
|---|---|
| `walk` | Walks forward, moving one leg at a time (most stable) |
| `trot` | Walks forward faster, diagonal legs move together |
| `turn left` | Spins 90° left on the spot, then stands still |
| `turn right 45` | Spins 45° right (any angle works), then stands still |
| `stop` | Stands still |
| `params` | Shows the current settings |
| `set <name> <value>` | Changes a setting live, e.g. `set shoulder_sweep 0.4` |

Settings you can change:

| Setting | Meaning | Default |
|---|---|---|
| `shoulder_home` | Shoulder angle when standing | 0.687 rad |
| `shoulder_sweep` | How far each leg swings → bigger = longer steps | 0.35 rad |
| `knee_home` | Knee angle when the foot is on the ground | -1.0 rad |
| `knee_lift` | Knee angle when the foot is in the air | -1.3 rad |
| `cycle_period` | Seconds per step cycle when walking → smaller = faster | 1.4 s |
| `turn_period` | Same, but when turning | 1.0 s |

Mouse in the 3D window: left-drag = rotate view, right-drag = move view, scroll = zoom.

---

## How the walking works

**One leg, one cycle** (repeats over and over):

```
  SWING (foot in the air, ~1/3 of the time)     STANCE (foot on the ground, ~2/3)
  knee bends up, leg swings forward              knee straight, leg pushes backward
                                                 → this push moves the body forward
```

The swing follows a smooth **Bezier curve**, so the leg moves smoothly instead of jerking.

**Four legs together** – all legs do the same cycle, just shifted in time:

| Gait | Leg order | Feel |
|---|---|---|
| Walk | one leg at a time | slow, very stable |
| Trot | diagonal pairs together (front-right + rear-left, then front-left + rear-right) | faster |

**Turning on the spot** – the sideways hip joints are locked, so the robot can't step
sideways. It turns like a tank instead:

```
  turn left:   left legs step BACKWARD,  right legs step FORWARD   → body spins left
  turn right:  left legs step FORWARD,   right legs step BACKWARD  → body spins right
```

While turning, the controller keeps checking which way the body points. When it has
rotated the requested angle, it stops and stands. Accuracy is about ±2.5°.

---

## How the RL works

**What RL controls** – 6 walking settings:

| # | Setting | Meaning |
|---|---|---|
| 1 | shoulder_home | standing shoulder angle |
| 2 | sweep_left | step length of the left legs |
| 3 | sweep_right | step length of the right legs |
| 4 | knee_home | knee angle on the ground |
| 5 | knee_lift | knee angle in the air |
| 6 | cycle_period | how fast the legs cycle |

**One try = one episode:**
1. RL picks the 6 settings.
2. The robot walks with them for up to 20 s in the simulation (`gym_class.py`).
3. It gets a **score**:
   - ➕ walking forward at about 0.3 m/s
   - ➕ staying alive (not falling)
   - ➖ drifting sideways or spinning
   - ➖ body tilting, bouncing up/down, wobbling
   - ➖ using a lot of energy
   - ➖ big penalty for falling over
4. RL adjusts its choices to get a higher score next time.

The algorithm is **PPO** (from the `stable-baselines3` library).

```bash
python -m sim.rl_ppo train                       # default 2000 tries, saves quadruped_bezier_ppo.zip
python -m sim.rl_ppo train --episodes 500        # fewer tries
python -m sim.rl_ppo test                        # no model: tries the middle value of every setting
python -m sim.rl_ppo test --model quadruped_bezier_ppo   # test the trained model
python -m sim.rl_ppo test --no-render            # without the 3D window
```

Every try prints the settings it used, whether the robot fell, its speed, and the
best settings found so far.

---

## The robot hardware

| Part | Details |
|---|---|
| Brain (microcontroller) | ESP32 |
| Motor driver | PCA9685 (controls up to 16 servos) |
| Motors | 12 servos, 3 per leg |
| Battery | LiPo |
| Planned onboard computer | NVIDIA Jetson Nano |

Each real leg has 3 motors: **sideways hip**, **shoulder**, **knee**. In the
simulation the sideways hip is **locked**, so only 8 motors move (shoulder + knee per leg).

Simulated sensors (not on the real robot yet): body tilt/rotation sensor (IMU) and
a foot-contact sensor on each leg.

---

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Docker setup + GitHub | ✅ Done |
| 2 | Robot model in the simulation | ✅ Done |
| 3 | Smooth leg curves (Bezier) + walk / trot | ✅ Done |
| 4 | Turn left / right on command | ✅ Done |
| 5 | RL to improve the walk | 🔄 In progress |
| 6 | Put the best RL settings into the controller | ⬜ Next |
| 7 | Run it on the real robot (ESP32) | ⬜ Later |
| 8 | Jetson Nano onboard computer | ⬜ Later |

---

*Bachelor's project — Robotics Engineering, Germany*
