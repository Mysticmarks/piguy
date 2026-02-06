#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/run"
LOG_PREFIX="[download-models]"

mkdir -p "${RUN_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ "${PIGUY_DOWNLOAD_TRANSFORMERS:-1}" == "1" ]]; then
  echo "${LOG_PREFIX} Restoring browser transformer assets"
  "${ROOT_DIR}/scripts/restore-models.sh"
fi

if [[ "${PIGUY_DOWNLOAD_WHISPER:-1}" == "1" ]]; then
  if [[ -x "${PYTHON_BIN}" ]]; then
    echo "${LOG_PREFIX} Prefetching Whisper tiny model"
    "${PYTHON_BIN}" - <<'PY'
import whisper
whisper.load_model("tiny")
print("whisper tiny ready")
PY
  else
    echo "${LOG_PREFIX} Skip Whisper prefetch: missing ${PYTHON_BIN}"
  fi
fi

if [[ "${PIGUY_DOWNLOAD_DIA2:-1}" == "1" ]]; then
  if [[ -x "${PYTHON_BIN}" ]]; then
    echo "${LOG_PREFIX} Prefetching Dia2 model"
    "${PYTHON_BIN}" - <<'PY'
import os
from dia2 import Dia2

repo = os.environ.get("DIA2_REPO", "nari-labs/Dia2-2B")
device = os.environ.get("DIA2_DEVICE")
dtype = os.environ.get("DIA2_DTYPE")

if not device:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
if not dtype:
    dtype = "bfloat16" if device == "cuda" else "float32"

Dia2.from_repo(repo, device=device, dtype=dtype)
print(f"dia2 ready: {repo}")
PY
  else
    echo "${LOG_PREFIX} Skip Dia2 prefetch: missing ${PYTHON_BIN}"
  fi
fi

touch "${RUN_DIR}/.models-ready"
echo "${LOG_PREFIX} Model prefetch complete"
