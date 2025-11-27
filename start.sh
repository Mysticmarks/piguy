#!/bin/bash
# Pi-Guy Dashboard Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Kill any existing server
pkill -f "python.*app.py" 2>/dev/null

# Start Flask server in background
./venv/bin/python app.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Open Chromium in kiosk mode
chromium --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --disable-restore-session-state http://localhost:5000

# When browser closes, kill server
kill $SERVER_PID 2>/dev/null
