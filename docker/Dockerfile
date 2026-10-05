# =============================================================================
# Mini-SPOT Simulation Environment
# =============================================================================
# What's inside:
#   - Python 3.11
#   - MuJoCo (physics simulation)
#   - NumPy, SciPy (math and kinematics)
#   - Matplotlib (plotting trajectories and debugging)
#   - PyYAML (reading robot_params.yaml)
#   - PySerial (ESP32 hardware communication)
#   - Pytest (testing IK, Bezier, gait scheduler)
# =============================================================================

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# SYSTEM PACKAGES
# Everything MuJoCo needs to run and display a window
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.11 \
    python3.11-dev \
    python3-pip \
    # OpenGL — required by MuJoCo renderer
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    # X11 — lets MuJoCo open a window on your screen
    libx11-6 \
    libxext6 \
    libxrender1 \
    libosmesa6-dev \
    # General utilities
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# -----------------------------------------------------------------------------
# PYTHON PACKAGES
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    # Physics simulation
    mujoco==3.1.6 \
    # Math and kinematics
    numpy==1.26.4 \
    scipy==1.13.0 \
    # Plotting (trajectory visualization, debugging)
    matplotlib==3.9.0 \
    # Config files (robot_params.yaml)
    PyYAML==6.0.1 \
    # ESP32 serial communication (hardware bridge, Session 7)
    pyserial==3.5 \
    # Testing
    pytest==8.2.2 \
    pytest-cov==5.0.0

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (sim/gym_class.py)
# CPU-only torch wheel: no CUDA in the container, and the default PyPI wheel
# drags in ~2 GB of CUDA libs on x86.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    gymnasium==0.29.1 \
    stable-baselines3==2.3.2

# One OpenMP thread: avoids torch/MuJoCo thread-pool clashes (segfaults)
ENV OMP_NUM_THREADS=1

# -----------------------------------------------------------------------------
# MUJOCO DISPLAY SETTINGS
# MUJOCO_GL=glfw    → opens a real window
# MUJOCO_GL=egl     → headless
# Override at runtime: docker run -e MUJOCO_GL=egl ...
# -----------------------------------------------------------------------------
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# -----------------------------------------------------------------------------
# WORKSPACE
# -----------------------------------------------------------------------------
WORKDIR /workspace
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1


ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} m_spot && \
    useradd -m -u ${UID} -g ${GID} m_spot

USER m_spot
CMD ["bash"]
