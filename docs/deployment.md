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

## CDN fallback behavior (local-first, offline-friendly)

Frontend vendor libraries are loaded from local static assets first:

- `/static/vendor/socket.io.min.js`
- `/static/vendor/transformers.min.js`

Remote CDN script fallback is **disabled by default** and only allowed when
`ALLOW_CDN_FALLBACK=true` is set in the runtime environment (or equivalent runtime config injection).

Expected behavior:

- `ALLOW_CDN_FALLBACK=false` (default): app runs in offline-first mode and never attempts remote CDN
  script fetches. Missing local assets fail fast with a clear console warning/error.
- `ALLOW_CDN_FALLBACK=true`: app may load remote CDN assets when local vendor files are unavailable,
  and the UI displays a visible warning banner that remote fallback is active and offline behavior may
  be degraded.

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
