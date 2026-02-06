#!/bin/bash
# Pi-Guy Dashboard Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/bootstrap.sh"

# Load environment variables
load_env_file "${SCRIPT_DIR}/.env"

# Audio capture device (override in .env or shell, examples: default, plughw:2,0)
export PIGUY_AUDIO_DEVICE="${PIGUY_AUDIO_DEVICE:-default}"
PIGUY_PORT="${PIGUY_PORT:-5000}"
PID_FILE="${SCRIPT_DIR}/run/piguy.pid"
mkdir -p "${SCRIPT_DIR}/run"

# Resolve venv and verify dependencies.
# Auto-install is allowed only when PIGUY_DEPENDENCY_MODE=dev.
VENV_DIR="$(ensure_venv_dependencies "${SCRIPT_DIR}")"

# Queue first-run model downloads (optional, background).
queue_model_downloads_once "${SCRIPT_DIR}"

# Kill existing tracked server if still valid for this repo
if [ -f "${PID_FILE}" ]; then
    OLD_PID="$(cat "${PID_FILE}" 2>/dev/null)"
    if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        OLD_CMDLINE="$(tr '\0' ' ' < "/proc/${OLD_PID}/cmdline" 2>/dev/null)"
        if [[ "${OLD_CMDLINE}" == *"${SCRIPT_DIR}"* ]] && [[ "${OLD_CMDLINE}" == *"app.py"* ]]; then
            kill "${OLD_PID}" 2>/dev/null
        fi
    fi
    rm -f "${PID_FILE}"
fi

# Start Flask server in background
"${VENV_DIR}/bin/python" app.py &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

# Wait for server to start
sleep 3

# Open Chromium in kiosk mode
BROWSER_BIN="$(find_chromium_browser)"

if [ -z "${BROWSER_BIN}" ]; then
    echo "No Chromium-compatible browser found. Set BROWSER_BIN to a browser executable."
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

${BROWSER_BIN} --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --disable-restore-session-state "http://localhost:${PIGUY_PORT}"

# When browser closes, kill server
if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null
fi
if [ -f "${PID_FILE}" ] && [ "$(cat "${PID_FILE}" 2>/dev/null)" = "${SERVER_PID}" ]; then
    rm -f "${PID_FILE}"
fi
