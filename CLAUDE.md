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
├── start-server.sh             # Backend-only launcher for systemd
├── start.sh                    # Interactive kiosk launcher (server + browser)
├── requirements.txt            # Python dependencies
├── pi-guy-dashboard.service    # Systemd backend service file
├── pi-guy-dashboard-kiosk.service # Optional desktop kiosk systemd unit
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
- **Living presence loop** - breathing/sway micro-motion + mood intensity decay to feel alive
- **Waveform mouth** - oscilloscope animation for TTS sync
- **Mini stats bar** at bottom showing CPU/Temp/Mem/Disk
- **Trigger buttons** on sides for testing expressions

## Running the Dashboard

### Desktop shortcut (recommended)
Double-click `Pi-Guy-Dashboard` on the desktop

### Manual launch
```bash
cd /home/mike/pi-01/Dashboard
# One-time autonomous build/setup (recommended):
./scripts/build.sh --profile all
# Runtime launch (offline-safe):
./start.sh
```

### Server only (no browser)
```bash
cd /home/mike/pi-01/Dashboard
./start-server.sh
# Then open http://localhost:5000 in any browser
```

## Environment variables
- `PIGUY_AUDIO_DEVICE` (default: `default`) controls microphone device used by `arecord`.
  - Example USB mic: `export PIGUY_AUDIO_DEVICE=plughw:2,0`
  - Example system default input: `export PIGUY_AUDIO_DEVICE=default`

## Dependencies
Dependency installation is intentionally separated from runtime startup.

### Profiles
- **core** (`requirements-core.txt`): Flask server + monitoring dependencies (no heavyweight speech runtimes).
- **speech** (`requirements-speech.txt`): Whisper/OpenAI/TTS extras, including Dia2.
- **all** (`requirements.txt`): installs both `core` and `speech` profiles.

### Provision dependencies explicitly
```bash
# Install core only
./scripts/install-deps.sh --profile core

# Install speech extras only
./scripts/install-deps.sh --profile speech

# Install everything (default for full feature set)
./scripts/install-deps.sh --profile all
```

### Autonomous build/setup helper
Use `scripts/build.sh` when you want a single command that validates prerequisites,
installs dependencies, and runs practical first-run setup steps.

```bash
# Default: install all deps + restore browser transformer assets
./scripts/build.sh

# Full setup including Whisper + Dia2 model prefetch
./scripts/build.sh --models

# Build dependencies only
./scripts/build.sh --skip-models
```

Runtime launch scripts no longer fetch packages in production mode, so startup works without internet/package index access as long as provisioning has already happened.

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



## Transformer.js model cache + fallback

The project now includes a model manifest at `static/models/transformers/manifest.json` used by `static/js/model-loader.js`.

- If model assets exist locally under `static/models/transformers/...`, the UI can use them.
- If local files are missing, it falls back to jsDelivr/unpkg CDN URLs from the manifest.
- Restore missing assets with:

```bash
./scripts/restore-models.sh
# or force re-download
./scripts/restore-models.sh --force
```

## Deployment Workflow (build/deploy vs runtime)

1. **Build/deploy phase (with package index access):** run `./scripts/install-deps.sh --profile <profile>` to create/update `./venv`.
2. **Ship/runtime phase (no network expected):** run `./start.sh` or `./start-face.sh`.
3. In runtime bootstrap, set `PIGUY_DEPENDENCY_MODE=prod` (default) to fail fast if venv deps are missing/stale.
4. Use `PIGUY_DEPENDENCY_MODE=dev` only when you explicitly want bootstrap to auto-install during development.

## CI Build Snapshots

Each push/PR now runs `.github/workflows/build-and-snapshot.yml`, which:
- Boots the Flask app in CI
- Captures dashboard and face PNG snapshots
- Uploads them as downloadable workflow artifacts (`piguy-snapshots-<run_number>`)

This gives you a visual snapshot for every build so behavior/expression changes are easy to review over time.

This decoupling keeps service startup deterministic and avoids runtime dependency on PyPI/network availability.

## Deployment Hardening (non-local)

Use environment variables to enable strict defaults when exposing Pi-Guy beyond localhost.

### Environment variables
- `PIGUY_ENV`: `dev` (default) or `prod`
- `PIGUY_DEPENDENCY_MODE`: `prod` (default fail-fast) or `dev` (allow auto-install in bootstrap)
- `PIGUY_DEPENDENCY_PROFILE`: `core`, `speech`, or `all` (default)
- `SECRET_KEY`: Flask secret key (required in `prod`)
- `PIGUY_API_KEY`: shared API key for control endpoints (required in `prod`)
- `PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS`: comma-separated Socket.IO allowed origins (required in `prod`)
- `PIGUY_API_CORS_ALLOWED_ORIGINS`: comma-separated allowed origins for `/api/*` HTTP requests (required in `prod`)
- `PIGUY_BIND_HOST`: bind host (defaults: `0.0.0.0` in `dev`, `127.0.0.1` in `prod`)
- `PIGUY_PORT`: bind port (default `5000`)

### Production startup example
```bash
export PIGUY_ENV=prod
export SECRET_KEY='replace-with-long-random-secret'
export PIGUY_API_KEY='replace-with-long-random-api-key'
export PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS='https://dashboard.example.com'
export PIGUY_API_CORS_ALLOWED_ORIGINS='https://dashboard.example.com'
export PIGUY_BIND_HOST=127.0.0.1
export PIGUY_PORT=5000
./venv/bin/python app.py
```


### SPA prototyping without build
- You can open `templates/index.html` directly (for example via `file://`) during design/prototyping.
- The UI auto-targets `http://localhost:5000` when loaded from `file://`.
- Override the backend base URL via query string, e.g. `index.html?apiBase=http://127.0.0.1:5000`.
- The selected `apiBase` is persisted in `localStorage` key `piguy-api-base`.

### Control API authentication
When `PIGUY_API_KEY` is set, **all** `/api/*` endpoints require `X-API-Key` (including chat, vision, realtime, stats, audio, and face/control APIs).

```bash
# Example: face control endpoint
curl -H "X-API-Key: $PIGUY_API_KEY" http://localhost:5000/api/mood/happy

# Example: chat endpoint
curl -H "X-API-Key: $PIGUY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}' \
  http://localhost:5000/api/chat
```

### Reverse proxy guidance (internet-accessible deployments)
- Terminate TLS at a reverse proxy (Nginx/Caddy/Traefik).
- Keep Pi-Guy bound to localhost (`PIGUY_BIND_HOST=127.0.0.1`) so it is not directly exposed.
- Forward WebSocket traffic for Socket.IO routes.
- Restrict allowed origins with `PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS` and `PIGUY_API_CORS_ALLOWED_ORIGINS` to your public dashboard domain(s).
- Use strong random values for `SECRET_KEY` and `PIGUY_API_KEY`.

## Notes
- Browser: Uses `chromium` command (Pi OS default)
- Kiosk mode: Press Alt+F4 to exit fullscreen
- Port: 5000 (change in app.py if needed)
- Face eyes will follow cursor, ready for camera tracking integration


## First-run model prefetch queue

To keep runtime self-contained, launchers now can queue model downloads on first run.

- `scripts/queue-model-downloads.sh`: one-time gate + optional prompt/background queue
- `scripts/download-required-models.sh`: downloads browser transformer assets and prefetches Whisper/Dia2 python models

Behavior is controlled with environment variables:

- `PIGUY_AUTO_MODEL_DOWNLOAD=ask|1|0` (default `ask`)
- `PIGUY_MODEL_PREFETCH_ENABLED=1|0` (default `1`)
- `PIGUY_DOWNLOAD_TRANSFORMERS=1|0` (default `1`)
- `PIGUY_DOWNLOAD_WHISPER=1|0` (default `1`)
- `PIGUY_DOWNLOAD_DIA2=1|0` (default `1`)

The queue runs in the background and writes logs to `run/model-download.log`.
