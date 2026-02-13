# Deployment Environment Requirements

Production service installs must provide explicit environment values and must not rely on
`app.py` development defaults.

## Required environment variables

Set these values in `/etc/piguy/piguy.env` (recommended) or in `${PIGUY_HOME}/.env`:

- `PIGUY_ENV=prod`
- `SECRET_KEY=<long-random-secret>`
- `PIGUY_API_KEY=<service-api-key>`
- `PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS=<comma-separated-allowed-origins>`
- `PIGUY_API_CORS_ALLOWED_ORIGINS=<comma-separated-allowed-origins>`

Example:

```env
PIGUY_ENV=prod
SECRET_KEY=replace-with-a-long-random-secret
PIGUY_API_KEY=replace-with-a-strong-api-key
PIGUY_SOCKETIO_CORS_ALLOWED_ORIGINS=https://dashboard.example.com
PIGUY_API_CORS_ALLOWED_ORIGINS=https://dashboard.example.com
```

## Frontend API key propagation

When `PIGUY_API_KEY` is enabled, browser-originated `/api/*` calls must send the same key via
`X-API-Key`. The dashboard frontend reads this key from one of these client-safe runtime sources:

1. `window.PiGuyRuntimeConfig.apiKey` (or `api_key`) when your reverse proxy/template injects a runtime config object.
2. `window.__PIGUY_CONFIG__.apiKey` (or `api_key`) if you use that global config convention.
3. Browser local storage key `piguy-api-key` (fallback for kiosk/dev setups).

`static/js/companion-config.js` normalizes these sources and exposes `PiGuyCompanionConfig.apiKey`,
which the dashboard fetch helpers then attach as `X-API-Key` for all `/api/*` requests. If no key
is present, requests behave exactly as before and send only `Content-Type: application/json`.

## Customize before install (required)

Before installing/enabling systemd units on a host, set concrete values for these fields:

- `User=`
- `Environment=PIGUY_HOME=`
- `WorkingDirectory=`
- optional local override path: `EnvironmentFile=-...`

For a direct edit workflow, update both `pi-guy-dashboard.service` and
`pi-guy-dashboard-kiosk.service` so they match your target host/user:

```ini
User=<target-linux-user>
Environment=PIGUY_HOME=/home/<target-linux-user>/piguy
WorkingDirectory=/home/<target-linux-user>/piguy
EnvironmentFile=-/home/<target-linux-user>/piguy/.env
```

For a template workflow, render concrete units from:

- `pi-guy-dashboard.service.template`
- `pi-guy-dashboard-kiosk.service.template`

using:

```bash
sudo scripts/install-systemd-services.sh \
  --user <target-linux-user> \
  --home /home/<target-linux-user>/piguy \
  --env-file /etc/piguy/piguy.env \
  --local-env-file /home/<target-linux-user>/piguy/.env
```

This script renders concrete values, writes units to `/etc/systemd/system`, runs
`systemctl daemon-reload`, and enables both services (unless `--no-enable` is passed).

## systemd services

`pi-guy-dashboard.service` and `pi-guy-dashboard-kiosk.service` are configured to:

- force `PIGUY_ENV=prod`
- read secure environment values from `/etc/piguy/piguy.env`
- optionally read `${PIGUY_HOME}/.env` for local overrides

## Production safety checks

`start-server.sh` exits with a clear error when:

- `PIGUY_ENV` is not exactly `prod`
- any required production variable above is missing

This ensures service templates do not silently fall back to development defaults.

## Dependency bootstrap

Install dependencies from locked manifests to satisfy runtime bootstrap checks:

```bash
scripts/install-deps.sh --profile all
```

For profile-specific installs, use `--profile core` or `--profile speech`. See
`docs/dependency-management.md` for lock-file regeneration and upgrade workflow.

## Health probes (`/api/health/liveness` and `/api/health/readiness`)

Pi-Guy exposes two dedicated health endpoints:

- `GET /api/health/liveness`
  - Minimal process check only.
  - Returns `200` with `{ "status": "ok", "alive": true, "pid": ... }` when the Flask process is running.
  - Does **not** validate model or TTS dependencies.

- `GET /api/health/readiness`
  - Structured dependency/config readiness checks with per-check machine-readable status:
    - `model_provider` (provider API reachability and chat capability path expectations)
    - `tts_dia2`, `tts_xtts`, `tts_piper` (backend availability checks)
    - `prod_config` (required prod-mode env/config presence)
  - Each check reports `status` as one of `pass | warn | fail` plus detail metadata.
  - Aggregate response fields:
    - `readiness`: `pass | warn | fail`
    - `ready`: boolean (`false` only when aggregate is `fail`)
  - HTTP behavior:
    - `200` for aggregate `pass` or `warn`
    - `503` for aggregate `fail`

### Kubernetes probe usage

Use liveness for restart safety and readiness for traffic gating.

```yaml
livenessProbe:
  httpGet:
    path: /api/health/liveness
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /api/health/readiness
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3
```

### systemd watchdog/monitoring usage

systemd itself does not directly consume HTTP probes, but operators can gate restarts/alerts using
`ExecStartPre` or timer-based checks against readiness:

```ini
ExecStartPre=/usr/bin/curl --fail --silent http://127.0.0.1:5000/api/health/readiness
```

Recommended pattern:

- Use process-level supervision (`Restart=on-failure`) for liveness.
- Use external polling (timer unit, node exporter textfile check, or service monitor) against
  `/api/health/readiness` to detect degraded dependencies before user-visible failures.
