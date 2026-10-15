#!/usr/bin/env bash
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Preparation only by default. No mode provisions Docker, models or viewers.
set -euo pipefail

DEMO_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEMO_UV_VERSION=0.11.8

demo_usage() {
    cat <<'EOF'
Usage: bash scripts/demo_install.sh [plan|system|environment] [--root PATH] [--apply]

  plan          Describe the staged installation; no changes (default).
  system        Print administrator package commands; never execute them.
  environment   Describe the user-owned environment. With --apply, install it.
  --root PATH   Existing, dedicated installation directory; required to apply.
  --apply       Explicitly authorize environment installation as the demo user.

Installation is preparation, not a running or secured public demo.
Docker/runtime privileges, viewers, model weights, datasets, demo accounts,
expiration, service startup and public access require separate steps.
EOF
}

demo_fail() { printf 'demo-install: %s\n' "$*" >&2; return 1; }

demo_plan() {
    cat <<'EOF'
No changes will be made without environment --apply.

1. system: print the Ubuntu/Debian prerequisite commands for an administrator.
   Review and run those commands separately; do not grant demo sudo access.
2. environment --root PATH --apply: as demo, install uv 0.11.8, managed Python
   3.12, the locked workspace dependencies, and Node/Corepack if required.
   This downloads software, including large ML libraries, but no model weights
   or datasets. It does not modify shell profiles or the system Python.
3. Stop for review: Docker/NVIDIA Container Toolkit and container privileges
   are NOT installed or configured by this script. OHIF, Slicer, QuPath and
   CVAT provisioning, model caching and service startup are separate approvals.

Suggested layout under the dedicated root:
  src/          transferred source (no existing workspace or credentials)
  env/          Python environment
  tools/        uv, managed Python and optional Node/Corepack
  cache/        dependency and future model/download caches
  workspace/    future application state (no server/database created here)
  tmp/          retained installer staging files

Do not use the upstream setup.sh for this staged flow: it can start Docker,
grant Docker-group access and prepare viewers. This script reuses selected
helpers only, without calling setup.sh's main function.
EOF
}

demo_load_helpers() {
    # setup.sh has a main guard; sourcing it defines helpers without installing.
    # shellcheck source=setup.sh
    source "$DEMO_SOURCE/setup.sh"
    platform_setup
}

demo_system_plan() {
    demo_load_helpers
    printf '%s\n' 'Review these commands; this script does not execute them:'
    printf '%s\n' 'sudo apt-get update'
    printf 'sudo apt-get install --no-remove --no-install-recommends'
    printf ' %q' "${SETUP_PACKAGES[@]}"
    printf '\n'
    printf '%s\n' 'Package-manager scripts may affect existing system services; review the APT proposal.'
}

demo_environment() {
    local root="$1" owner tool package uv_name uv_version
    [[ "$(id -un)" == demo && "$(id -u)" != 0 ]] ||
        demo_fail 'Run environment installation as demo, without sudo.'
    [[ "$root" == /* && -d "$root" && ! -L "$root" ]] ||
        demo_fail '--root must be an existing absolute directory, not a symlink.'
    root="$(cd "$root" && pwd -P)"
    [[ "$root" != / && "$root" != "$HOME" && "$root" != "$DEMO_SOURCE" ]] ||
        demo_fail 'Choose a dedicated installation root, not /, your home or the source directory.'
    owner="$(stat -c '%u' "$root")"
    [[ "$owner" == "$(id -u)" && -w "$root" ]] ||
        demo_fail 'The installation root must be owned by and writable by demo.'
    for tool in curl tar xz git sha256sum; do
        command -v "$tool" >/dev/null || demo_fail "Missing $tool; review the system step first."
    done
    demo_load_helpers
    for package in "${SETUP_PACKAGES[@]}"; do
        installed "$package" || demo_fail "Missing $package; review the system step first."
    done
    [[ -f "$DEMO_SOURCE/uv.lock" && -f "$DEMO_SOURCE/.python-version" ]] ||
        demo_fail 'The transferred source is incomplete.'

    # Keep bulky dependencies and caches under the explicitly selected root.
    umask 077
    export UV_PROJECT_ENVIRONMENT="$root/env"
    export UV_CACHE_DIR="$root/cache/uv"
    export UV_PYTHON_INSTALL_DIR="$root/tools/python"
    export UV_PYTHON_BIN_DIR="$root/tools/bin"
    export UV_PYTHON=3.12
    export UV_TOOL_DIR="$root/tools/uv-tools"
    export UV_TOOL_BIN_DIR="$root/tools/bin"
    export XDG_CACHE_HOME="$root/cache"
    export TMPDIR="$root/tmp"
    export MONAILABEL_DATA_DIR="$root/workspace"
    export MONAILABEL_CACHE_DIR="$root/cache/monailabel"
    export MONAILABEL_TOOLS_DIR="$root/tools"
    export HF_HOME="$root/cache/huggingface"
    export TORCH_HOME="$root/cache/torch"
    export PYTHONNOUSERSITE=1
    export UV_NO_MODIFY_PATH=1
    mkdir -p "$TMPDIR" "$root/tools/uv"
    SETUP_TEMP="$(mktemp -d "$TMPDIR/install.XXXXXXXX")"
    SETUP_ENV="$UV_PROJECT_ENVIRONMENT"
    printf 'Installer staging retained at %s\n' "$SETUP_TEMP"

    if [[ ! -x "$root/tools/uv/uv" ]]; then
        fetch "https://astral.sh/uv/$DEMO_UV_VERSION/install.sh" "$SETUP_TEMP/uv-install.sh"
        UV_UNMANAGED_INSTALL="$root/tools/uv" sh "$SETUP_TEMP/uv-install.sh"
    fi
    read -r uv_name uv_version _ <<< "$("$root/tools/uv/uv" --version)"
    [[ "$uv_name" == uv && "$uv_version" == "$DEMO_UV_VERSION" ]] ||
        demo_fail "Expected uv $DEMO_UV_VERSION; inspect the existing tools/uv installation."
    export PATH="$root/tools/uv:$SETUP_ENV/bin:$PATH"
    cd "$DEMO_SOURCE"
    uv python install 3.12
    ensure_python
    ensure_node
    printf '%s\n' 'Software environment prepared. No server, viewer or model has been started.'
    printf '%s\n' 'Stop for review before provisioning Docker, viewers, models or public access.'
}

demo_main() {
    local phase=plan root='' apply=0
    if (($#)) && [[ "$1" != --* && "$1" != -h ]]; then phase="$1"; shift; fi
    while (($#)); do
        case "$1" in
            --root)
                (($# >= 2)) || { demo_fail '--root needs a value.'; return 1; }
                root="$2"; shift 2 ;;
            --apply) apply=1; shift ;;
            --help|-h) demo_usage; return ;;
            *) demo_fail "Unknown option: $1"; return 1 ;;
        esac
    done
    case "$phase" in plan|system|environment) ;; *) demo_fail "Unknown phase: $phase"; return 1 ;; esac
    if ((apply)) && [[ "$phase" != environment ]]; then
        demo_fail '--apply is only supported for environment; system commands require separate review.'
        return 1
    fi
    case "$phase" in
        plan) demo_plan ;;
        system) demo_system_plan ;;
        environment)
            if ((apply)); then
                [[ -n "$root" ]] || { demo_fail '--root is required with --apply.'; return 1; }
                demo_environment "$root"
            else
                demo_plan
            fi ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then demo_main "$@"; fi
