#!/bin/bash
# Pi-Guy Face Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing server
pkill -f "python.*app.py" 2>/dev/null

# Start Flask server in background
./venv/bin/python app.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Open Chromium (fullscreen but NOT kiosk - can close with Alt+F4)
chromium --start-fullscreen --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --disable-restore-session-state http://localhost:5000/face

# When browser closes, kill server
kill $SERVER_PID 2>/dev/null
