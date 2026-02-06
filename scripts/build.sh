#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PIGUY_DEPENDENCY_PROFILE:-all}"
SETUP_MODELS="auto"
QUEUE_MODELS=0

usage() {
    cat <<USAGE
Usage: scripts/build.sh [options]

Autonomous project build/setup entrypoint that validates prerequisites,
installs Python dependencies, and performs pragmatic first-run setup.

Options:
  --profile core|speech|all   Dependency profile (default: all)
  --models                    Download/restore required models now
  --skip-models               Skip model setup entirely
  --queue-models              Queue model downloads in background after build
  -h, --help                  Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="${2:-}"
            shift 2
            ;;
        --models)
            SETUP_MODELS="now"
            shift
            ;;
        --skip-models)
            SETUP_MODELS="skip"
            shift
            ;;
        --queue-models)
            QUEUE_MODELS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[build] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${PROFILE}" in
    core|speech|all) ;;
    *)
        echo "[build] Unsupported profile '${PROFILE}'. Use core, speech, or all." >&2
        exit 1
        ;;
esac

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[build] Missing prerequisite: ${cmd}" >&2
        exit 1
    fi
}

require_cmd bash
require_cmd python3
require_cmd sha256sum

if [[ "${SETUP_MODELS}" != "skip" ]]; then
    require_cmd curl
fi

cd "${ROOT_DIR}"

echo "[build] Installing dependencies (profile=${PROFILE})"
"${ROOT_DIR}/scripts/install-deps.sh" --profile "${PROFILE}"

if [[ "${SETUP_MODELS}" == "now" ]]; then
    echo "[build] Restoring browser transformer assets"
    "${ROOT_DIR}/scripts/restore-models.sh"
    echo "[build] Prefetching optional runtime models"
    "${ROOT_DIR}/scripts/download-required-models.sh"
elif [[ "${SETUP_MODELS}" == "auto" ]]; then
    echo "[build] Running pragmatic default model setup (transformers only)"
    PIGUY_DOWNLOAD_WHISPER=0 PIGUY_DOWNLOAD_DIA2=0 \
        "${ROOT_DIR}/scripts/download-required-models.sh"
else
    echo "[build] Skipping model setup"
fi

if [[ "${QUEUE_MODELS}" == "1" ]]; then
    echo "[build] Queueing background model downloads"
    PIGUY_AUTO_MODEL_DOWNLOAD=1 "${ROOT_DIR}/scripts/queue-model-downloads.sh" --force --auto
fi

echo "[build] Build/setup complete"
