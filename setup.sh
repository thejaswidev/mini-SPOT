#!/bin/bash
# =============================================================================
# Mini-SPOT — setup.sh
# =============================================================================
# Run this once to set up and launch your development environment.
#
# Usage:
#   chmod +x setup.sh    (only once)
#   ./setup.sh
#
# What it does:
#   1. Checks Docker is installed
#   2. Creates .env from .env.example if it doesn't exist
#   3. Allows Docker to use your display (X11)
#   4. Builds the Docker image
#   5. Starts the container
#   6. Verifies MuJoCo is working
# =============================================================================

set -e  # stop immediately if any command fails

# -----------------------------------------------------------------------------
# COLORS — just for readable output
# -----------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # no color

print_step()  { echo -e "\n${YELLOW}▶ $1${NC}"; }
print_ok()    { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# -----------------------------------------------------------------------------
# STEP 1 — Check Docker is installed
# -----------------------------------------------------------------------------
print_step "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed."
    echo "Install it from https://docs.docker.com/get-docker/ then run this script again."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker is installed but not running."
    echo "Start Docker Desktop (macOS/Windows) or run: sudo systemctl start docker (Linux)"
    exit 1
fi

print_ok "Docker is running."

# -----------------------------------------------------------------------------
# STEP 2 — Create .env from .env.example if it doesn't exist
# -----------------------------------------------------------------------------
print_step "Checking .env file..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # always run from repo root regardless of where you call the script from

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_ok ".env created from .env.example."
        echo "  Default DISPLAY=:0 is set. Edit .env if your display is different."
    else
        print_error ".env.example not found. Make sure you're in the mini-SPOT repo root."
        exit 1
    fi
else
    print_ok ".env already exists. Skipping."
fi

# -----------------------------------------------------------------------------
# STEP 3 — X11 display access (Linux only)
# macOS and Windows handle this differently — see README.md
# -----------------------------------------------------------------------------
print_step "Setting up display access for MuJoCo viewer..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xhost +local:docker &> /dev/null && print_ok "X11 access granted to Docker." \
        || echo "  Warning: xhost failed. MuJoCo viewer may not open. Try: xhost +local:docker"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  macOS detected. Make sure XQuartz is running and you've run: xhost +localhost"
else
    echo "  Windows detected. Make sure VcXsrv is running with 'Disable access control' checked."
fi

# -----------------------------------------------------------------------------
# STEP 4 — Build the Docker image
# -----------------------------------------------------------------------------
print_step "Building Docker image..."

docker compose -f docker/docker-compose.yml build

print_ok "Docker image built successfully."

# -----------------------------------------------------------------------------
# STEP 5 — Start the container
# -----------------------------------------------------------------------------
print_step "Starting container..."

docker compose -f docker/docker-compose.yml up -d

print_ok "Container is running."

# -----------------------------------------------------------------------------
# STEP 6 — Verify MuJoCo works
# -----------------------------------------------------------------------------
print_step "Verifying MuJoCo installation..."

MUJOCO_VERSION=$(docker compose -f docker/docker-compose.yml exec -T mini-spot \
    python -c "import mujoco; print(mujoco.__version__)" 2>&1)

if [[ "$MUJOCO_VERSION" == 3* ]]; then
    print_ok "MuJoCo $MUJOCO_VERSION is working inside the container."
else
    print_error "MuJoCo check failed. Output: $MUJOCO_VERSION"
    echo "  Try running manually: docker compose -f docker/docker-compose.yml exec mini-spot bash"
    exit 1
fi

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Mini-SPOT environment is ready!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "To get a shell inside the container:"
echo "  docker compose -f docker/docker-compose.yml exec mini-spot bash"
echo ""
echo "To stop the container:"
echo "  docker compose -f docker/docker-compose.yml down"
echo ""
