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
