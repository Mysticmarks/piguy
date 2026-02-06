#!/bin/bash
# Pi-Guy Server Launcher (systemd-safe: backend only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/bootstrap.sh"

# Load environment variables
load_env_file "${SCRIPT_DIR}/.env"

# Audio capture device (override in .env or shell, examples: default, plughw:2,0)
export PIGUY_AUDIO_DEVICE="${PIGUY_AUDIO_DEVICE:-default}"

# Create venv and install dependencies if needed
VENV_DIR="$(ensure_venv_dependencies "${SCRIPT_DIR}")"

# Run backend in foreground so supervisor restart behavior tracks Python process exits.
exec "${VENV_DIR}/bin/python" app.py
