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

ensure_venv_dependencies() {
    local script_dir="$1"

    local venv_dir="${VENV_DIR:-${script_dir}/venv}"
    local req_file="${script_dir}/requirements.txt"
    local hash_file="${venv_dir}/.requirements.sha256"
    local need_install=0
    local req_hash=""

    if [ ! -x "${venv_dir}/bin/python" ]; then
        python3 -m venv "${venv_dir}"
        need_install=1
    fi

    if [ -f "${req_file}" ]; then
        req_hash="$(sha256sum "${req_file}" | awk '{print $1}')"
        if [ ! -f "${hash_file}" ] || [ "$(cat "${hash_file}")" != "${req_hash}" ]; then
            need_install=1
        fi
    fi

    if [ "${need_install}" -eq 1 ]; then
        "${venv_dir}/bin/python" -m pip install --upgrade pip
        if [ -f "${req_file}" ]; then
            "${venv_dir}/bin/pip" install -r "${req_file}"
            echo "${req_hash}" > "${hash_file}"
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
