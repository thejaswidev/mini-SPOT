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
- **RL, stage 2** = a small neural network watches the IMU 50× per second and slightly reshapes each
  leg's Bézier curve, so the robot copes with bumps, slopes and pushes.

---

## What each file does

### The files you actually run

| File | What it is | When to use it |
|---|---|---|
| `sim/gait_controller.py` | **Drive the robot yourself.** Opens a 3D window; you type `walk`, `turn left`, `stop`… | To watch the robot move and test gaits by hand |
| `sim/rl_ppo.py` | **RL stage 1.** Learns the best 6 walking settings | To improve the base walk |
| `sim/rl_residual.py` | **RL stage 2.** Trains / tests the network that reshapes the curve (terrain, slopes, pushes) | To make the walk robust |
| `sim/rl_command.py` | **One network for every move (paper architecture).** Train / test / **drive** it: forward, backward, strafe, turn | The main RL from now on |
| `sim/mujoco_env.py` | **Just show the robot standing.** The simplest viewer | To check the robot model loads |
| `tests/` | **Automatic checks.** Turning angle, circle size, and the RL stage 2 environment (terrain, slopes, zero change = original curve) | After changing code, to make sure nothing broke |

### The files the others use (you don't run these directly)

| File | What it is |
|---|---|
| `sim/bezier_helpers.py` | **The gait math.** Says where every leg should be at every moment (the smooth leg curves and the timing between legs). Shared by both the manual controller and RL. |
| `sim/gym_class.py` | **RL stage 1 training ground.** Takes a set of walking settings, lets the robot walk with them for 20 s in the simulation, and gives back a **score**. |
| `sim/residual_env.py` | **RL stage 2 training ground.** Runs the Bézier gait, lets the network reshape each leg's curve slightly every 1/50 s, on random bumps and slopes with pushes, and scores how well it keeps walking. |
| `assets/scene_terrain.xml` | **Bumpy world** for stage 2 (the bumps are re-randomized every episode). |
| `sim/bezier3d.py` | **The paper's 3D Bézier gait.** Turns a command (forward / sideways speed, turn rate) into foot positions in 3D, and a fast exact IK turns those into the 3 joint angles of each leg. |
| `sim/command_env.py` | **Training ground for the one-network-for-all-moves policy.** |
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

### RL stage 2: the network slightly reshapes the Bézier curve (`rl_residual.py`)

Stage 1 picks the settings **once** — it cannot react to a push, a bump or a slope. Stage 2
follows the D²-GMBC paper (Rahme et al. 2020): a small neural network runs **all the time**
(50× per second) and **bends each leg's Bézier curve a little**:

```
  IMU (tilt + rotation speed) ─┬──►  neural network  ──►  5 small curve changes per leg
  gait clock + last action ────┘                                   │
                                                                   ▼
             Bézier gait (stage 1 settings)  ──►  reshaped curve  ──►  joint targets  ──►  servos
```

The 5 changes per leg — all **0 = the original curve**, and all kept small:

| Change | What it does | Paper | Max |
|---|---|---|---|
| clearance | lift the foot higher / lower in the swing (step over bumps) | ψ | ±50 % of the lift |
| depth | in stance, push the foot slightly *into* the ground, or hold it above (finds ground that is lower / higher than expected) | δ | ±0.05 rad |
| shoulder | shift the whole loop forward / back (lean into a slope) | foot offset x | ±0.1 rad |
| knee | leg a bit longer / shorter (keep the body level on a slope) | foot offset z | ±0.1 rad |
| sideways hip | small sideways foot shift (side slopes, sideways pushes) | foot offset y | ±0.1 rad |

- **Inputs** are only the BNO055 IMU plus the gait clock, so the same network can run on the robot.
- **Training happens only in simulation**, every episode on a new random world
  (`assets/scene_terrain.xml`): bumps 0–6 cm, uphill/downhill ±15°, side slopes ±8° (pushes
  are off by default), and different friction, body mass, motor strength, sensor noise and servo delay.
  Slopes are made by tilting gravity, which is the same physics as standing on an incline.
- **Score:** ➕ walking at ~0.15 m/s, body level and at the right height above the ground, not
  falling; ➖ large and jerky curve changes (it should stay close to the Bézier gait).

```bash
python -m sim.rl_residual train                     # 3 M steps, 8 parallel sims, saves models/residual_ppo.zip
python -m sim.rl_residual eval                      # Bézier only vs Bézier + network on flat / bumps / slopes / pushes
python -m sim.rl_residual eval --scenarios uphill15 bumps6 --episodes 20
python -m sim.rl_residual watch --scenario uphill15 # watch it in 3D (on a Mac: .venv/bin/mjpython -m sim.rl_residual watch ...)
python -m sim.rl_residual eval --model models/residual_ckpt_2000000_steps   # test a checkpoint (saved every 250 k steps)
python -m sim.rl_residual train --gait-params models/best_params.json   # build on stage 1's best settings
```

Scenarios: `flat`, `bumps4`, `bumps6`, `uphill10`, `uphill15`, `downhill15`, `side8`, `push80`,
`mixed` (everything at once, like training). Stopping training with Ctrl+C saves to
`models/residual_ppo_interrupted` — the finished model is not overwritten.

**Result** (3 M training steps; 16 test episodes of 10 s per row, randomized robots not seen in
training; distance = forward progress):

| Scenario | Bézier only | Bézier + network |
|---|---|---|
| flat | 16/16 stay up, 1.57 m | 16/16, 1.28 m |
| bumps 4 cm | 16/16, 1.32 m | 16/16, 1.09 m |
| bumps 6 cm | 16/16, 1.11 m | 16/16, 0.73 m |
| uphill 10° | 16/16, 0.76 m | 16/16, **0.96 m** |
| **uphill 15°** | **5/16**, −0.16 m | **12/16**, +0.12 m |
| downhill 15° | 16/16, 1.84 m | 16/16, 1.59 m |
| side slope 8° | 16/16, 1.22 m | 16/16, 1.20 m |
| **mixed** (all at once) | **12/16**, 0.78 m | **16/16**, 0.83 m |

The network fixes the plain gait's weak spot — steep uphill — and never falls on the mixed
terrain, but it walks more carefully (15–35 % slower) where the plain gait was already fine.

### One network for all moves (`rl_command.py`) — the paper's architecture

This follows the D²-GMBC paper (Fig. 2) exactly. The Bézier curve now describes **where the foot
goes in 3D**, and a **command** says which way to go:

```
  command (forward/back speed, sideways speed, turn rate) ───────────────┐
                                                                         ▼
  IMU + gait clock + command ──► network ──► ψ, δ  (curve shape) ──►  3D Bézier gait (sim/bezier3d.py)
                                        └──► Δf    (small foot offsets) ──(+)──► IK ──► 12 servos
```

- **One network learns every move:** forward, backward, strafe left / right, turn left / right
  and stand. During training the command changes every 3–6 s, so it also learns switching.
- **Network outputs (14), all small, 0 = the plain paper gait:** ψ foot clearance (0.04 ± 0.02 m),
  δ stance depth (0.005 ± 0.005 m), and a foot offset per leg in x, y, z (± 1.5 cm).
- **Inputs:** IMU (tilt + rotation speed), gait clock, the command, last action — like the paper.
- **Trained on** the same random bumps, slopes and robot changes as stage 2.
- **Score:** ➕ moving at the commanded speed / turn rate, body level, not falling; ➖ big or jerky changes.

The plain paper gait (no network) already follows every command on flat ground — the network's
job is to keep that working on bumps and slopes:

| Command | Asked | Plain paper gait got |
|---|---|---|
| forward / backward | ±0.15 m/s | +0.14 / −0.18 m/s |
| strafe left / right | ±0.10 m/s | +0.10 / −0.10 m/s |
| turn left / right | ±0.5 rad/s | +0.57 / −0.58 rad/s |

```bash
python -m sim.rl_command drive --plain            # drive the plain paper gait in 3D (no training needed)
python -m sim.rl_command train                    # train the network (5 M steps, 8 parallel sims)
python -m sim.rl_command drive                    # drive the trained network: w a s d = move, q e = turn, x = stop
python -m sim.rl_command eval                     # plain gait vs gait + network for every command (flat)
python -m sim.rl_command eval --scenario mixed    # same on bumps + slopes
```

On a Mac use `.venv/bin/mjpython` for `drive` and `.venv/bin/python` for the rest.

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
| 6 | RL stage 2: curve changes for bumps, slopes and pushes (sim) | 🔄 In progress |
| 7 | Real slopes / stairs in the simulation (not only tilted gravity) | ⬜ Next |
| 8 | Run it on the real robot (Jetson runs the network, ESP32 the servos) | ⬜ Later |

---

*Bachelor's project — Robotics Engineering, Germany*
