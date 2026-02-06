#!/bin/bash

# Shared launcher bootstrap helpers

load_env_file() {
    local env_file="${1:-.env}"

    if [ -f "${env_file}" ]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
}

resolve_requirements_file() {
    local script_dir="$1"
    local profile="${PIGUY_DEPENDENCY_PROFILE:-all}"

    case "${profile}" in
        core)
            echo "${script_dir}/requirements-core.txt"
            ;;
        speech)
            echo "${script_dir}/requirements-speech.txt"
            ;;
        all)
            echo "${script_dir}/requirements.txt"
            ;;
        *)
            echo "Unsupported PIGUY_DEPENDENCY_PROFILE='${profile}'. Use core, speech, or all." >&2
            return 1
            ;;
    esac
}

ensure_venv_dependencies() {
    local script_dir="$1"

    local venv_dir="${VENV_DIR:-${script_dir}/venv}"
    local req_file
    req_file="$(resolve_requirements_file "${script_dir}")" || return 1

    local dep_mode="${PIGUY_DEPENDENCY_MODE:-prod}"
    local hash_file="${venv_dir}/.$(basename "${req_file}").sha256"
    local req_hash=""

    if [ ! -x "${venv_dir}/bin/python" ]; then
        if [ "${dep_mode}" = "dev" ]; then
            echo "[bootstrap] Creating virtualenv at ${venv_dir} (dev mode)."
            python3 -m venv "${venv_dir}" || return 1
        else
            echo "[bootstrap] Missing virtualenv at ${venv_dir}." >&2
            echo "[bootstrap] Run scripts/install-deps.sh before starting in ${dep_mode} mode." >&2
            return 1
        fi
    fi

    if [ ! -f "${req_file}" ]; then
        echo "[bootstrap] Requirements file not found: ${req_file}" >&2
        return 1
    fi

    req_hash="$(sha256sum "${req_file}" | awk '{print $1}')"

    if [ ! -f "${hash_file}" ] || [ "$(cat "${hash_file}")" != "${req_hash}" ]; then
        if [ "${dep_mode}" = "dev" ]; then
            echo "[bootstrap] Installing/updating dependencies from ${req_file} (dev mode)."
            "${venv_dir}/bin/python" -m pip install --upgrade pip || return 1
            "${venv_dir}/bin/pip" install -r "${req_file}" || return 1
            echo "${req_hash}" > "${hash_file}"
        else
            echo "[bootstrap] Dependency set is missing or stale for ${req_file}." >&2
            echo "[bootstrap] Run scripts/install-deps.sh --profile ${PIGUY_DEPENDENCY_PROFILE:-all} before starting in ${dep_mode} mode." >&2
            return 1
        fi
    fi

    echo "${venv_dir}"
}

find_chromium_browser() {
    local browser_bin="${BROWSER_BIN:-}"

    if [ -z "${browser_bin}" ]; then
        if command -v chromium >/dev/null 2>&1; then
            browser_bin="chromium"
        elif command -v chromium-browser >/dev/null 2>&1; then
            browser_bin="chromium-browser"
        elif command -v google-chrome >/dev/null 2>&1; then
            browser_bin="google-chrome"
        fi
    fi

    echo "${browser_bin}"
}
