#!/bin/bash
# Pi-Guy Server Launcher (systemd-safe: backend only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/bootstrap.sh"

# Load environment variables
load_env_file "${SCRIPT_DIR}/.env"

if [ "${PIGUY_ENV:-}" != "prod" ]; then
    echo "ERROR: start-server.sh is intended for production service usage and requires PIGUY_ENV=prod." >&2
    echo "Set PIGUY_ENV=prod in the systemd unit or environment file before starting the service." >&2
    exit 1
fi

required_prod_env_vars=(
    SECRET_KEY
    PIGUY_API_KEY
    PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS
    PIGUY_API_CORS_ALLOWED_ORIGINS
)

for required_var in "${required_prod_env_vars[@]}"; do
    if [ -z "${!required_var:-}" ]; then
        echo "ERROR: Required production environment variable '${required_var}' is not set." >&2
        echo "Define it in /etc/piguy/piguy.env or ${SCRIPT_DIR}/.env before starting the service." >&2
        exit 1
    fi
done

# Audio capture device (override in .env or shell, examples: default, plughw:2,0)
export PIGUY_AUDIO_DEVICE="${PIGUY_AUDIO_DEVICE:-default}"

# Create venv and install dependencies if needed
VENV_DIR="$(ensure_venv_dependencies "${SCRIPT_DIR}")"

# Queue first-run model downloads (optional, background).
queue_model_downloads_once "${SCRIPT_DIR}"

# Run backend in foreground so supervisor restart behavior tracks Python process exits.
exec "${VENV_DIR}/bin/python" app.py
