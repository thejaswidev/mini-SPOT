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
| IMU | Basic IMU |
| Onboard compute (planned) | NVIDIA Jetson Nano |


4 legs × 3 joints = 12 servos total. Each leg has full 3-DOF — Hip Abduction/Adduction, Hip Flexion/Extension, Knee Flexion/Extension.

---

## Roadmap

| Session | Focus | Status |
|---|---|---|
| 1 | Docker environment + GitHub setup | 🔄 In progress |
| 2 | MJCF robot model in MuJoCo | ⬜ Pending |
| 3 | Inverse Kinematics solver | ⬜ Pending |
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
chmod +x setup.sh
./setup.sh
```

**macOS:**
Install [XQuartz](https://www.xquartz.org) first, then:
```bash
xhost +localhost
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x setup.sh
./setup.sh
```

**Windows:**
Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), launch XLaunch with "Disable access control" checked, then:
```bash
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x setup.sh
./setup.sh
```

---

## Repo Structure

```
mini-SPOT/
├── docker/
│   ├── Dockerfile              ← Python 3.11, MuJoCo, all dependencies
│   └── docker-compose.yml      ← container config, volume mounts, display
├── assets/
│   └── mini_spot.xml           ← MuJoCo MJCF robot model
├── sim/
│   ├── ik_solver.py            ← IKSolver — foot position → joint angles
│   ├── bezier_trajectory.py    ← BezierTrajectoryGenerator — swing foot path
│   ├── gait_scheduler.py       ← GaitScheduler — trot/crawl timing
│   ├── locomotion_controller.py← top-level controller
│   └── mujoco_env.py           ← MuJoCo simulation runner
├── hardware/
│   ├── esp32/
│   │   ├── servo_driver.py     ← MicroPython: drives PCA9685
│   │   └── calibrate.py        ← servo calibration tool
│   └── serial_bridge.py        ← laptop → ESP32 serial bridge
├── tests/
│   ├── test_ik.py
│   ├── test_bezier.py
│   └── test_gait.py
├── setup.sh                    ← one-command environment setup

```

---

## Tech Stack

| Tool | Role |
|---|---|
| Python 3.11 | Primary language |
| MuJoCo | Physics simulation |
| MJCF XML | Robot model format |
| NumPy / SciPy | Kinematics and math |
| Matplotlib | Visualization and debugging |
| Docker | Reproducible environment |
| ESP32 + MicroPython | Low-level servo control |


---

## Architecture

```
VelocityCommand (vx, vy, yaw)
    └── GaitScheduler        → phase and swing/stance per leg
          └── BezierTrajectory → foot target position (x, y, z)
                └── IKSolver   → joint angles (θ_HAA, θ_HFE, θ_KFE)
                      └── MuJoCo actuators / ESP32 servos
```

Control loop: physics at 1000Hz, controller at 100Hz.



*Bachelor's project — Robotics Engineering, Germany*
