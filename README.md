# Mini-SPOT

A small 4-legged robot dog (inspired by Boston Dynamics Spot), built as a
Bachelor's project in Robotics Engineering.

Right now everything runs in a **physics simulation** (MuJoCo) on the computer.
The robot can walk, trot, turn left/right, and walk in circles on command. Reinforcement
learning (RL) is being added to find a smoother, better walking style.

> This README describes the `rl-gait` branch.

---

## The 30-second version

- The robot has **4 legs**, each with **3 motors**: a **sideways hip**, a **shoulder** (swings the
  leg forward/back) and a **knee** (bends the leg). 4 × 3 = **12 motors**. The walking gait uses
  the shoulder and knee; the sideways hips are used by the RL corrections.
- Walking = moving each leg in a repeating loop: **lift it, swing it forward, put it down, push back**.
  Doing this with the 4 legs at different times makes the body move forward.
- **Turning** = left legs and right legs push in opposite directions, so the body spins on the spot.
- **RL, stage 1** = the computer tries thousands of different walking settings (step length, speed,
  knee bend…) and keeps the ones that make the robot walk fast, straight and steady.
- **RL, stage 2** = a small neural network watches the IMU and foot sensors 50× per second and adds
  small corrections to every motor, so the robot recovers when it is pushed.

---

## What each file does

### The files you actually run

| File | What it is | When to use it |
|---|---|---|
| `sim/gait_controller.py` | **Drive the robot yourself.** Opens a 3D window; you type `walk`, `turn left`, `stop`… | To watch the robot move and test gaits by hand |
| `sim/rl_ppo.py` | **RL stage 1.** Learns the best 6 walking settings | To improve the base walk |
| `sim/rl_residual.py` | **RL stage 2.** Trains / tests the correction network (push recovery) | To make the walk robust |
| `sim/mujoco_env.py` | **Just show the robot standing.** The simplest viewer | To check the robot model loads |
| `tests/` | **Automatic checks.** Turning angle, circle size, and the RL stage 2 environment | After changing code, to make sure nothing broke |

### The files the others use (you don't run these directly)

| File | What it is |
|---|---|
| `sim/bezier_helpers.py` | **The gait math.** Says where every leg should be at every moment (the smooth leg curves and the timing between legs). Shared by both the manual controller and RL. |
| `sim/gym_class.py` | **RL stage 1 training ground.** Takes a set of walking settings, lets the robot walk with them for 20 s in the simulation, and gives back a **score**. |
| `sim/residual_env.py` | **RL stage 2 training ground.** Runs the Bézier gait, adds the network's corrections every 1/50 s, pushes the robot around and scores how well it stays up and keeps walking. |
| `3dbezierhelpers.py` | **Alternative gait (from the D²-GMBC paper)** — Bézier curves for the foot position in 3D + inverse kinematics. Standalone demo for now. |
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
        ┌────────────────────┼─────────────────────────┬──────────────────────────────┐
        │                    │                         │                              │
 sim/mujoco_env.py   sim/gait_controller.py      sim/gym_class.py                sim/residual_env.py
 (just stands)       (YOU give commands)         (scores a set of settings)      (scores the corrections)
                             │                    ▲ used by sim/rl_ppo.py         ▲ used by sim/rl_residual.py
                             │                    │   RL STAGE 1                  │   RL STAGE 2
                             └──────── all use ───┴───────────┬───────────────────┘
                                                              ▼
                                                    sim/bezier_helpers.py
                                                    (where each leg goes, when)
```

**Three pieces:**

1. **Manual** – `gait_controller.py`: fixed rules. You give commands, it walks / turns / circles.
2. **RL stage 1** – `rl_ppo.py` + `gym_class.py`: finds the best 6 walking settings (open loop:
   picked once, then the robot just walks).
3. **RL stage 2** – `rl_residual.py` + `residual_env.py`: a network that reacts to the sensors
   50× per second and corrects the gait (closed loop: handles pushes).

Turning is **only manual** on purpose. RL only learns forward walking, so turning
can't mess up the walking training.

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
| `circle left` | Walks in a circle to the left (1 m radius) until you type `stop` |
| `circle right 2` | Walks in a circle to the right with a 2 m radius (smallest: 0.6 m) |
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

**Turning on the spot** – the gait keeps the sideways hips straight, so the robot doesn't
step sideways. It turns like a tank instead:

```
  turn left:   left legs step BACKWARD,  right legs step FORWARD   → body spins left
  turn right:  left legs step FORWARD,   right legs step BACKWARD  → body spins right
```

While turning, the controller keeps checking which way the body points. When it has
rotated the requested angle, it stops and stands. Once it has settled it measures again and
makes a small correction turn if needed. Accuracy is about ±2°.

**Walking in a circle** – the robot trots forward, but the legs on the **inside** of the
circle take **shorter steps** than the outside legs, so the path bends (like a car where
the inner wheels turn slower). The controller keeps measuring how big the circle it is
actually walking is, and adjusts the inside step length until it matches the radius you
asked for (accurate to a few cm). A 1 m circle takes a bit over a minute.

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
python -m sim.rl_ppo train                       # default 20000 tries, saves quadruped_bezier_ppo.zip
python -m sim.rl_ppo train --episodes 500        # fewer tries
python -m sim.rl_ppo test                        # no model: tries the middle value of every setting
python -m sim.rl_ppo test --model quadruped_bezier_ppo   # test the trained model
python -m sim.rl_ppo test --no-render            # without the 3D window
```

Training runs 6 simulations in parallel, logs every try to `models/episodes.csv` and saves
the best settings to `models/best_params.json`.

### RL stage 2: corrections for push recovery (`rl_residual.py`)

Stage 1 picks the settings **once** — it cannot react if someone pushes the robot. Stage 2 adds a
small neural network that runs **all the time** (50× per second):

```
  IMU (tilt + rotation speed) ─┬──►  neural network  ──►  small correction (delta) for each of the 12 motors
  gait clock + last action ────┘                                   │
                                                                   ▼
             Bézier gait (stage 1 settings)  ──►  joint targets  + delta  ──►  servos
```

- **Inputs** are only the BNO055 IMU (no foot sensors) plus the gait clock, so the same network can
  run on the hardware later.
- **Outputs** are corrections of at most ±0.25 rad, added on top of the Bézier gait — the gait
  does the walking, the network only fixes mistakes. It can also use the sideways hips.
- **Training happens only in the simulation**, with random pushes (30–110 N), and with friction,
  body mass, motor strength, sensor noise and servo delay changed every episode. A network that
  copes with all of that in simulation has a much better chance on the real robot
  ("sim-to-real"). Training on the real robot itself would need thousands of falls.
- **Score:** ➕ walking at ~0.15 m/s, staying upright and at the right height, not falling;
  ➖ big corrections and jerky corrections (it should stay close to the Bézier gait).

```bash
python -m sim.rl_residual train                     # 3 M steps (~20 min), 8 parallel sims, saves models/residual_ppo.zip
python -m sim.rl_residual eval                      # Bézier only vs Bézier + network, pushes of 60 / 80 / 100 N
python -m sim.rl_residual watch --push 80           # watch it in 3D (on a Mac: .venv/bin/mjpython -m sim.rl_residual watch)
python -m sim.rl_residual eval --model models/residual_ckpt_2000000_steps   # test a checkpoint (saved every 250 k steps)
python -m sim.rl_residual train --gait-params models/best_params.json   # build on stage 1's best settings
```

Stopping training with Ctrl+C saves to `models/residual_ppo_interrupted` — the finished model is not overwritten.

**First result** (3 M training steps ≈ 20 min on a laptop; 20 test episodes of 10 s per row,
a push every 1.5–3 s, randomized robots not seen in training):

| Push | Bézier only | Bézier + network |
|---|---|---|
| none | 20/20 stay up, 1.55 m | 20/20 stay up, 1.29 m |
| 60 N | 20/20, 1.32 m | 18/20, 1.23 m |
| **80 N** | **4/20**, 0.74 m | **13/20**, 1.01 m |
| 100 N | 0/20, 0.53 m | 4/20, 0.63 m |

The network more than triples survival at 80 N and starts to handle 100 N. It still walks ~15 %
slower without pushes and occasionally falls at 60 N where the plain gait doesn't — results vary
between training runs. Next steps: longer training, rougher terrain, tuning the speed reward.

---

## The robot hardware

| Part | Details |
|---|---|
| Brain (microcontroller) | ESP32 |
| Motor driver | PCA9685 (controls up to 16 servos) |
| Motors | 12 servos, 3 per leg |
| Battery | LiPo |
| Planned onboard computer | NVIDIA Jetson Nano |

Each leg has 3 motors: **sideways hip**, **shoulder**, **knee** — all 12 are in the simulation.
The Bézier gait drives shoulder + knee and keeps the sideways hips straight; the stage 2 network
may move all 12.

Sensors: BNO055 IMU (tilt, rotation speed) — simulated now,
on the robot later.

---

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Docker setup + GitHub | ✅ Done |
| 2 | Robot model in the simulation | ✅ Done |
| 3 | Smooth leg curves (Bezier) + walk / trot | ✅ Done |
| 4 | Turn left / right and walk in circles on command | ✅ Done |
| 5 | RL stage 1: best walking settings | 🔄 In progress |
| 6 | RL stage 2: corrections for push recovery (sim) | 🔄 In progress |
| 7 | Rougher terrain in training (bumps, slopes) | ⬜ Next |
| 8 | Run it on the real robot (Jetson runs the network, ESP32 the servos) | ⬜ Later |

---

*Bachelor's project — Robotics Engineering, Germany*
