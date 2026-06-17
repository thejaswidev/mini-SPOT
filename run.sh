#!/bin/bash
# =============================================================================
# Mini-SPOT — run.sh
# Starts the simulation container and drops you into a bash shell.
# Container is automatically deleted when you exit (--rm).
# =============================================================================

# Allow X11 connections from Docker (lets MuJoCo open a window)
xhost +local:docker

# Run container — drops you straight into bash
# --rm means container is deleted when you exit, no stuck containers ever
docker compose -f docker/docker-compose.yml run --rm mini-spot bash

# Revoke X11 access when done
xhost -local:docker
