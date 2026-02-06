#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/venv}"
PROFILE="all"

usage() {
    cat <<USAGE
Usage: scripts/install-deps.sh [--profile core|speech|all]

Installs Python dependencies into the project virtual environment and records
hashes used by runtime bootstrap checks.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            PROFILE="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[install-deps] Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip

install_profile() {
    local req_file="$1"
    local hash_file="${VENV_DIR}/.$(basename "${req_file}").sha256"
    local req_hash

    if [ ! -f "${req_file}" ]; then
        echo "[install-deps] Missing requirements file: ${req_file}" >&2
        exit 1
    fi

    echo "[install-deps] Installing ${req_file}"
    "${VENV_DIR}/bin/pip" install -r "${req_file}"
    req_hash="$(sha256sum "${req_file}" | awk '{print $1}')"
    echo "${req_hash}" > "${hash_file}"
}

case "${PROFILE}" in
    core)
        install_profile "${SCRIPT_DIR}/requirements-core.txt"
        ;;
    speech)
        install_profile "${SCRIPT_DIR}/requirements-speech.txt"
        ;;
    all)
        install_profile "${SCRIPT_DIR}/requirements.txt"
        ;;
    *)
        echo "Unsupported --profile '${PROFILE}'. Use core, speech, or all." >&2
        exit 1
        ;;
esac

echo "[install-deps] Done."
