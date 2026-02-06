#!/bin/bash
# Pi-Guy Dashboard Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Audio capture device (override in .env or shell, examples: default, plughw:2,0)
export PIGUY_AUDIO_DEVICE="${PIGUY_AUDIO_DEVICE:-default}"

# Create venv and install dependencies if needed
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/venv}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
HASH_FILE="${VENV_DIR}/.requirements.sha256"
NEED_INSTALL=0

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}"
    NEED_INSTALL=1
fi

if [ -f "${REQ_FILE}" ]; then
    REQ_HASH="$(sha256sum "${REQ_FILE}" | awk '{print $1}')"
    if [ ! -f "${HASH_FILE}" ] || [ "$(cat "${HASH_FILE}")" != "${REQ_HASH}" ]; then
        NEED_INSTALL=1
    fi
fi

if [ "${NEED_INSTALL}" -eq 1 ]; then
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    if [ -f "${REQ_FILE}" ]; then
        "${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"
        echo "${REQ_HASH}" > "${HASH_FILE}"
    fi
fi

# Kill any existing server
pkill -f "python.*app.py" 2>/dev/null

# Start Flask server in background
"${VENV_DIR}/bin/python" app.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Open Chromium in kiosk mode
BROWSER_BIN="${BROWSER_BIN:-}"
if [ -z "${BROWSER_BIN}" ]; then
    if command -v chromium >/dev/null 2>&1; then
        BROWSER_BIN="chromium"
    elif command -v chromium-browser >/dev/null 2>&1; then
        BROWSER_BIN="chromium-browser"
    elif command -v google-chrome >/dev/null 2>&1; then
        BROWSER_BIN="google-chrome"
    fi
fi

if [ -z "${BROWSER_BIN}" ]; then
    echo "No Chromium-compatible browser found. Set BROWSER_BIN to a browser executable."
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

${BROWSER_BIN} --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --disable-restore-session-state http://localhost:5000

# When browser closes, kill server
kill $SERVER_PID 2>/dev/null
