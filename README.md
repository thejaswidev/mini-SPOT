# Mini-SPOT

A miniature quadruped robot inspired by Boston Dynamics Spot.

The goal is a fully autonomous legged robot: simulation, kinematics, gait control, and real hardware deployment. 

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | ESP32 |
| Servo controller | PCA9685 (I2C, 16-channel PWM) |
| Servos | 12 total — 3 per leg (HAA, HFE, KFE joints) |
| Battery | LiPo |
| IMU | Basic IMU (simulation only for now) |
| Onboard compute (planned) | NVIDIA Jetson Nano |


4 legs × 3 joints = 12 servos total. Each leg has full 3-DOF — Hip Abduction/Adduction (HAA), Hip Flexion/Extension (HFE), Knee Flexion/Extension (KFE).

HAA joints are currently locked in simulation — 8 active DOF (HFE + KFE per leg). This simplifies IK to a clean 2D problem per leg.

---

## Roadmap

| Session | Focus | Status |
|---|---|---|
| 1 | Docker environment + GitHub setup | ✅ Complete |
| 2 | MJCF robot model in MuJoCo | ✅ Complete |
| 3 | Inverse Kinematics solver | 🔄 In progress |
| 4 | Bezier trajectory generator | ⬜ Pending |
| 5 | Gait scheduler (trot) | ⬜ Pending |
| 6 | Full locomotion controller — Mini-SPOT walks | ⬜ Pending |
| 7 | ESP32 hardware bridge | ⬜ Pending |
| 8 | Jetson Nano integration | ⬜ Pending |

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- Git

### Setup

**Linux:**
```bash
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

**macOS:**
Install [XQuartz](https://www.xquartz.org) first, then:
```bash
xhost +localhost
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

**Windows:**
Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), launch XLaunch with "Disable access control" checked, then:
```bash
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

### Run the simulation

```bash
# Inside the container
python sim/mujoco_env.py
```

MuJoCo viewer opens with Mini-SPOT standing on a checkered ground plane.
Use left mouse to orbit, right mouse to pan, scroll to zoom.

### Drive the robot (walk / trot / turn)

```bash
python sim/gait_controller.py        # macOS outside Docker: mjpython sim/gait_controller.py
```

Type commands in the terminal: `walk`, `trot`, `turn left`, `turn right 45`
(rotate in place by that many degrees, default 90, then stand), `stop`,
`params`, `set <param> <value>`.

### Reinforcement learning (PPO tunes the Bezier gait parameters)

```bash
python -m sim.rl_ppo train                 # trains, saves quadruped_bezier_ppo.zip
python -m sim.rl_ppo test --model quadruped_bezier_ppo
```

### Tests

```bash
python -m pytest tests
```

---

## Repo Structure

```
mini-SPOT/
├── run.sh                      ← one-command start (X11 + container)
├── docker/
│   ├── Dockerfile              ← Python 3.11, MuJoCo, all dependencies
│   └── docker-compose.yml      ← container config, volume mounts, display
├── assets/
│   ├── mini_spot.xml           ← MuJoCo MJCF robot model (pure robot description)
│   ├── scene.xml               ← simulation world (ground, lighting, skybox)
│   └── CAD_Files/
│       └── STL_combined/       ← mesh files for visual geometry
├── sim/
│   ├── mujoco_env.py           ← MuJoCo simulation runner
```

---

## Robot Model

The MJCF model (`assets/mini_spot.xml`) describes the full robot:

- Rectangular torso with freejoint (6 DOF in world)
- 4 legs × 3 bodies (hip, thigh, calf) with real STL mesh geometry
- 8 active position actuators (HFE + KFE per leg, HAA locked)
- IMU sensor site on trunk (accelerometer + gyro)
- Foot force sensor sites on each calf (contact detection)
- Joint limits matching real servo range: HFE [-0.785, 3.14] rad, KFE [-2.44, -0.916] rad
- Keyframes: `start` (crouched stand) and `home` (extended)

The scene file (`assets/scene.xml`) includes the robot and adds the world:
- Checkered ground plane with proper friction
- Skybox gradient
- Directional lighting with shadows

---

## Tech Stack

| Tool | Role |
|---|---|
| Python 3.11 | Primary language |
| MuJoCo 3.1.6 | Physics simulation |
| MJCF XML | Robot model format |
| NumPy / SciPy | Kinematics and math |
| Matplotlib | Visualization and debugging |
| Docker | Reproducible environment |
| ESP32 + MicroPython | Low-level servo control |
| NVIDIA Jetson Nano | Onboard compute |

---

## Architecture

```
VelocityCommand (vx, vy, yaw)
    └── GaitScheduler        → phase and swing/stance per leg
          └── BezierTrajectory → foot target position (x, z)
                └── IKSolver   → joint angles (θ_HFE, θ_KFE)
                      └── MuJoCo actuators / ESP32 servos
```

Physics runs at 1000Hz. Controller runs at 100Hz (every 10 physics steps).

---

## Sensors (Simulation)

All sensors are simulation-only

| Sensor | Location | Data | Hardware plan |
|---|---|---|---|
| Accelerometer | Trunk | Linear acceleration (m/s²) | Wire IMU to ESP32 via I2C |
| Gyroscope | Trunk | Angular velocity (rad/s) | Wire IMU to ESP32 via I2C |
| Foot force | Each calf | 3-axis contact force (N) | Estimate from joint torques |

---

*Bachelor's project — Robotics Engineering, Germany*

