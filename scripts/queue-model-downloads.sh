#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/run"
PID_FILE="${RUN_DIR}/model-download.pid"
QUEUE_MARKER="${RUN_DIR}/.models-queue-initialized"
LOG_FILE="${RUN_DIR}/model-download.log"

mkdir -p "${RUN_DIR}"

AUTO_MODE="${PIGUY_AUTO_MODEL_DOWNLOAD:-ask}" # ask|1|0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --auto)
      AUTO_MODE=1
      shift
      ;;
    --skip)
      AUTO_MODE=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${FORCE}" != "1" && -f "${QUEUE_MARKER}" ]]; then
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
    touch "${QUEUE_MARKER}"
    exit 0
  fi
fi

should_queue=0
if [[ "${AUTO_MODE}" == "1" ]]; then
  should_queue=1
elif [[ "${AUTO_MODE}" == "ask" && -t 0 ]]; then
  read -r -p "Queue required model downloads in background now? [Y/n] " answer
  answer="${answer:-Y}"
  if [[ "${answer}" =~ ^[Yy]$ ]]; then
    should_queue=1
  fi
fi

if [[ "${should_queue}" == "1" ]]; then
  nohup "${ROOT_DIR}/scripts/download-required-models.sh" >"${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "[model-queue] Started background model download job (pid $(cat "${PID_FILE}"))."
  echo "[model-queue] Log: ${LOG_FILE}"
else
  echo "[model-queue] Skipping model prefetch."
  echo "[model-queue] Set PIGUY_AUTO_MODEL_DOWNLOAD=1 to enable automatic first-run downloads."
fi

touch "${QUEUE_MARKER}"
