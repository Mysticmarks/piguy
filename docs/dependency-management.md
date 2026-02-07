# Dependency Management

Runtime dependencies are intentionally version-pinned and installed through lock manifests
for reproducible deployments.

## Files

- `requirements-core.txt`: pinned runtime packages needed to run the web app.
- `requirements-speech.txt`: pinned runtime speech/TTS packages.
- `requirements.txt`: composition file for full runtime profile.
- `requirements-core.lock.txt`, `requirements-speech.lock.txt`, `requirements.lock.txt`:
  lock manifests consumed by installers/bootstrap.

## Install flow

Use the installer script so production/bootstrap checks align with lock hashes:

```bash
scripts/install-deps.sh --profile core
scripts/install-deps.sh --profile speech
scripts/install-deps.sh --profile all
```

The bootstrap helper (`scripts/bootstrap.sh`) validates lock-file hashes and asks you to rerun
`install-deps.sh` when dependencies drift.

## CI enforcement

CI runs:

- `scripts/check-runtime-requirements-pinned.py` to reject unpinned runtime entries.
- `scripts/generate-lockfiles.sh` + `git diff --exit-code ...` to ensure lock manifests are
  regenerated and committed with every dependency change.

## Intentional upgrades

When bumping runtime dependencies:

1. Edit pins in `requirements-core.txt` and/or `requirements-speech.txt`.
2. Regenerate lock manifests:
   ```bash
   scripts/generate-lockfiles.sh
   ```
3. Reinstall from locks:
   ```bash
   scripts/install-deps.sh --profile all
   ```
4. Validate locally:
   ```bash
   python -m compileall app.py listen.py speak.py
   pytest -q
   ```
5. Commit both the pin changes and lock file updates together.
