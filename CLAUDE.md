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

## Environment variables
- `PIGUY_AUDIO_DEVICE` (default: `default`) controls microphone device used by `arecord`.
  - Example USB mic: `export PIGUY_AUDIO_DEVICE=plughw:2,0`
  - Example system default input: `export PIGUY_AUDIO_DEVICE=default`

## Dependencies
### Core (required)
- Flask
- Flask-SocketIO
- psutil
- simple-websocket
- requests
- dia2 (required for `/api/speak` local TTS)

### Speech transcription modes
- **Local Whisper mode (required for `/api/listen` and `listen.py` local transcription):**
  - `openai-whisper`
- **OpenAI API fallback mode (optional, only needed for `listen.py --api`):**
  - `openai` (plus `OPENAI_API_KEY` environment variable)

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


## Deployment Hardening (non-local)

Use environment variables to enable strict defaults when exposing Pi-Guy beyond localhost.

### Environment variables
- `PIGUY_ENV`: `dev` (default) or `prod`
- `SECRET_KEY`: Flask secret key (required in `prod`)
- `PIGUY_API_KEY`: shared API key for control endpoints (required in `prod`)
- `PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS`: comma-separated Socket.IO allowed origins (required in `prod`)
- `PIGUY_BIND_HOST`: bind host (defaults: `0.0.0.0` in `dev`, `127.0.0.1` in `prod`)
- `PIGUY_PORT`: bind port (default `5000`)

### Production startup example
```bash
export PIGUY_ENV=prod
export SECRET_KEY='replace-with-long-random-secret'
export PIGUY_API_KEY='replace-with-long-random-api-key'
export PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS='https://dashboard.example.com'
export PIGUY_BIND_HOST=127.0.0.1
export PIGUY_PORT=5000
./venv/bin/python app.py
```

### Control API authentication
When `PIGUY_API_KEY` is set, control endpoints require `X-API-Key`:

```bash
curl -H "X-API-Key: $PIGUY_API_KEY" http://localhost:5000/api/mood/happy
```

### Reverse proxy guidance (internet-accessible deployments)
- Terminate TLS at a reverse proxy (Nginx/Caddy/Traefik).
- Keep Pi-Guy bound to localhost (`PIGUY_BIND_HOST=127.0.0.1`) so it is not directly exposed.
- Forward WebSocket traffic for Socket.IO routes.
- Restrict allowed origins with `PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS` to your public dashboard domain(s).
- Use strong random values for `SECRET_KEY` and `PIGUY_API_KEY`.

## Notes
- Browser: Uses `chromium` command (Pi OS default)
- Kiosk mode: Press Alt+F4 to exit fullscreen
- Port: 5000 (change in app.py if needed)
- Face eyes will follow cursor, ready for camera tracking integration
