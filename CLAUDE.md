# Pi-Guy Dashboard

A sci-fi themed system monitoring dashboard for Raspberry Pi 5, with an animated face mode.

## Overview
- **Type**: Web app (Flask + HTML/CSS/JS)
- **Dashboard URL**: http://localhost:5000
- **Face URL**: http://localhost:5000/face
- **Style**: Futuristic sci-fi with glowing effects

## Files
```
/home/mike/pi-01/Dashboard/
├── app.py                      # Flask server with WebSocket stats + face API
├── venv/                       # Python virtual environment
├── templates/
│   ├── index.html              # Dashboard UI (gauges, styling, JS)
│   └── face.html               # Animated face with eyes + waveform mouth
├── start.sh                    # Launch script (starts server + browser)
├── requirements.txt            # Python dependencies
├── pi-guy-dashboard.service    # Systemd service file (optional)
└── CLAUDE.md                   # This file
```

## Features

### Dashboard Mode (/)
- **4 gauges**: CPU, Temperature, Memory, Disk
- **Real-time updates** via WebSocket (1 second refresh)
- **Gradient colors**: Green → Yellow → Red based on value
- **Scale numbers** around gauge arcs
- **Glow effects** and animated scan lines
- **Warning/danger states** with pulsing animations

### Face Mode (/face)
- **Animated eyes** that follow cursor (ready for camera tracking)
- **Expressions**: neutral, happy, sad, angry, thinking, surprised
- **Realistic blinking** with random intervals
- **Idle behavior** - eyes look around when inactive
- **Waveform mouth** - oscilloscope animation for TTS sync
- **Mini stats bar** at bottom showing CPU/Temp/Mem/Disk
- **Trigger buttons** on sides for testing expressions

## Running the Dashboard

### Desktop shortcut (recommended)
Double-click `Pi-Guy-Dashboard` on the desktop

### Manual launch
```bash
cd /home/mike/pi-01/Dashboard
./start.sh
```

### Server only (no browser)
```bash
cd /home/mike/pi-01/Dashboard
./venv/bin/python app.py
# Then open http://localhost:5000 in any browser
```

## Dependencies
- Flask
- Flask-SocketIO
- psutil
- simple-websocket

Installed in `./venv/` virtual environment.

## Customization

### Colors
Edit `:root` CSS variables in `templates/index.html`:
- `--blue`, `--red` for accent colors
- `--dark-bg`, `--panel-bg` for backgrounds

### Gauge gradient
Edit `getGradientColor()` function in the `<script>` section to change green→yellow→red progression.

### Add new gauges
1. Add stat collection in `app.py` `get_system_stats()`
2. Add SVG gauge in `templates/index.html`
3. Add update logic in the `socket.on('stats_update')` handler

## Face API

Control the face programmatically from other scripts (e.g., LLM/TTS integration):

```bash
# Set mood
curl http://localhost:5000/api/mood/happy
curl http://localhost:5000/api/mood/sad
curl http://localhost:5000/api/mood/angry
curl http://localhost:5000/api/mood/thinking
curl http://localhost:5000/api/mood/surprised
curl http://localhost:5000/api/mood/neutral

# Trigger blink
curl http://localhost:5000/api/blink

# Control talking animation
curl http://localhost:5000/api/talk/start
curl http://localhost:5000/api/talk/stop

# Make eyes look at position (x, y from 0.0 to 1.0)
curl "http://localhost:5000/api/look?x=0.8&y=0.2"
```

### WebSocket Events (for real-time control)
From your TTS/LLM process, emit these events:
- `set_mood` - `{mood: 'happy'}`
- `blink` - triggers a blink
- `start_talking` / `stop_talking` - controls waveform mouth
- `look_at` - `{x: 0.5, y: 0.5}` for eye position

## Face Keyboard Controls
- `1-6` - Switch moods (neutral, happy, sad, angry, thinking, surprised)
- `B` - Manual blink
- `Space` (hold) - Activate talking animation

## Notes
- Browser: Uses `chromium` command (Pi OS default)
- Kiosk mode: Press Alt+F4 to exit fullscreen
- Port: 5000 (change in app.py if needed)
- Face eyes will follow cursor, ready for camera tracking integration
