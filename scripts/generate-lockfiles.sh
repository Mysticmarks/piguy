#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${SCRIPT_DIR}/scripts/check-runtime-requirements-pinned.py"

cat > "${SCRIPT_DIR}/requirements-core.lock.txt" <<'LOCK'
# Locked runtime dependencies for core profile.
# Source of truth for requested packages lives in requirements-core.txt.
-r requirements-core.txt
LOCK

cat > "${SCRIPT_DIR}/requirements-speech.lock.txt" <<'LOCK'
# Locked runtime dependencies for speech profile.
# Source of truth for requested packages lives in requirements-speech.txt.
-r requirements-speech.txt
LOCK

cat > "${SCRIPT_DIR}/requirements.lock.txt" <<'LOCK'
# Locked runtime dependencies for full runtime profile.
# Source of truth for requested packages lives in requirements.txt.
-r requirements.txt
LOCK

echo "Regenerated lock manifests: requirements-core.lock.txt, requirements-speech.lock.txt, requirements.lock.txt"
