#!/bin/bash
# Pi-Guy Dashboard Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/bootstrap.sh"

# Load environment variables
load_env_file "${SCRIPT_DIR}/.env"

# Create venv and install dependencies if needed
VENV_DIR="$(ensure_venv_dependencies "${SCRIPT_DIR}")"

# Kill any existing server
pkill -f "python.*app.py" 2>/dev/null

# Start Flask server in background
"${VENV_DIR}/bin/python" app.py &
SERVER_PID=$!

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
    --disable-restore-session-state http://localhost:5000

# When browser closes, kill server
kill $SERVER_PID 2>/dev/null
