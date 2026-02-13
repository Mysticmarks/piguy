#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install-systemd-services.sh [options]

Render and install systemd unit files from templates, then enable services.

Options:
  --user USER             Linux user for service User= (default: pi)
  --home PATH             PIGUY_HOME path (default: /home/pi/piguy)
  --env-file PATH         Required env file path (default: /etc/piguy/piguy.env)
  --local-env-file PATH   Optional local override env file path (default: <home>/.env)
  --systemd-dir PATH      systemd unit directory (default: /etc/systemd/system)
  --no-enable             Do not run systemctl enable
  -h, --help              Show this help
USAGE
}

PIGUY_USER="pi"
PIGUY_HOME="/home/pi/piguy"
PIGUY_ENV_FILE="/etc/piguy/piguy.env"
PIGUY_LOCAL_ENV_FILE=""
SYSTEMD_DIR="/etc/systemd/system"
ENABLE_SERVICES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      PIGUY_USER="$2"
      shift 2
      ;;
    --home)
      PIGUY_HOME="$2"
      shift 2
      ;;
    --env-file)
      PIGUY_ENV_FILE="$2"
      shift 2
      ;;
    --local-env-file)
      PIGUY_LOCAL_ENV_FILE="$2"
      shift 2
      ;;
    --systemd-dir)
      SYSTEMD_DIR="$2"
      shift 2
      ;;
    --no-enable)
      ENABLE_SERVICES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PIGUY_LOCAL_ENV_FILE" ]]; then
  PIGUY_LOCAL_ENV_FILE="${PIGUY_HOME}/.env"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

render_unit() {
  local template="$1"
  local destination="$2"

  sed \
    -e "s|__PIGUY_USER__|${PIGUY_USER}|g" \
    -e "s|__PIGUY_HOME__|${PIGUY_HOME}|g" \
    -e "s|__PIGUY_ENV_FILE__|${PIGUY_ENV_FILE}|g" \
    -e "s|__PIGUY_LOCAL_ENV_FILE__|${PIGUY_LOCAL_ENV_FILE}|g" \
    "$template" > "$destination"
}

mkdir -p "$SYSTEMD_DIR"

render_unit \
  "${REPO_ROOT}/pi-guy-dashboard.service.template" \
  "${SYSTEMD_DIR}/pi-guy-dashboard.service"
render_unit \
  "${REPO_ROOT}/pi-guy-dashboard-kiosk.service.template" \
  "${SYSTEMD_DIR}/pi-guy-dashboard-kiosk.service"

systemctl daemon-reload

if [[ "$ENABLE_SERVICES" -eq 1 ]]; then
  systemctl enable pi-guy-dashboard.service pi-guy-dashboard-kiosk.service
fi

echo "Installed systemd units in ${SYSTEMD_DIR}"
echo "  User=${PIGUY_USER}"
echo "  PIGUY_HOME=${PIGUY_HOME}"
echo "  EnvironmentFile=${PIGUY_ENV_FILE}"
echo "  Local override EnvironmentFile=${PIGUY_LOCAL_ENV_FILE}"
