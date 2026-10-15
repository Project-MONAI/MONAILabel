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

# Standalone administrator bootstrap: never source service-user-owned code.
set -euo pipefail

bootstrap_fail() { printf 'demo-bootstrap: %s\n' "$*" >&2; return 1; }

bootstrap_plan() {
    cat <<'EOF'
Private demo host bootstrap for Ubuntu 22.04 x86_64.
Default: show this plan without changing anything.

After review, copy this script to /root/monailabel-demo-bootstrap.sh with
root ownership and mode 0700. Run that protected copy as root with:
  --apply --allow-docker-root-access

Installs missing system/viewer libraries, Docker Engine/Compose and NVIDIA
Container Toolkit from official package repositories. Configures the GPU
runtime and starts Docker; refuses to restart it with running containers.
Adds the existing demo account to Docker's ROOT-EQUIVALENT group and enables
its systemd user services after logout. This is for a trusted private host,
not anonymous visitor isolation. Does not grant sudo or change SSH settings.

Does not replace GPU drivers, reboot, remove packages, change router/firewall
rules, start MONAI Label or publish application ports. Docker may configure
its normal bridge/firewall rules. Docker data uses its system defaults;
application/model data is configured separately under the installation root.
EOF
}

bootstrap_installed() {
    [[ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null)" == installed ]]
}

bootstrap_install_missing() {
    local package missing=()
    for package in "$@"; do
        if ! bootstrap_installed "$package"; then missing+=("$package"); fi
    done
    if ((${#missing[@]})); then
        apt-get install -y --no-remove --no-install-recommends "${missing[@]}"
    fi
}

bootstrap_fetch() {
    curl --fail --location --show-error --silent --retry 3 \
        --connect-timeout 30 --max-time 300 --proto '=https' --proto-redir '=https' \
        "$1" -o "$2"
}

bootstrap_publish() {
    local source="$1" target="$2"
    if [[ -e "$target" || -L "$target" ]]; then
        if [[ -L "$target" ]] || ! cmp -s "$source" "$target"; then
            bootstrap_fail "Existing repository file differs: $target. Review it; not overwritten."
        fi
    else
        install -o root -g root -m 0644 "$source" "$target"
    fi
}

bootstrap_apply() {
    [[ "$EUID" == 0 ]] || bootstrap_fail 'Use the administrator-owned copy with sudo.'
    [[ "$(readlink -f "${BASH_SOURCE[0]}")" == /root/monailabel-demo-bootstrap.sh ]] ||
        bootstrap_fail 'First copy this script to /root/monailabel-demo-bootstrap.sh.'
    [[ "$(stat -c '%u:%a' /root/monailabel-demo-bootstrap.sh)" == 0:700 ]] ||
        bootstrap_fail 'The protected script must be root-owned with mode 0700.'
    export PATH=/usr/sbin:/usr/bin:/sbin:/bin
    export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
    unset DOCKER_HOST DOCKER_CONTEXT
    umask 022
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 && "$(uname -m)" == x86_64 ]] ||
        bootstrap_fail 'This reviewed bootstrap targets Ubuntu 22.04 x86_64 only.'
    [[ "$(id -u demo)" != 0 ]] || bootstrap_fail 'An existing non-root demo account is required.'
    nvidia-smi -L
    local package staging running
    for package in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc; do
        if bootstrap_installed "$package"; then
            bootstrap_fail "Conflicting package $package is installed; no automatic removal."
        fi
    done
    if command -v docker >/dev/null && systemctl is-active --quiet docker; then
        running="$(docker --host unix:///var/run/docker.sock ps -q)"
        [[ -z "$running" ]] || bootstrap_fail 'Running containers found; no changes made.'
    fi
    staging="$(mktemp -d /root/monailabel-bootstrap.XXXXXXXX)"
    printf 'Bootstrap staging retained at %s\n' "$staging"
    apt-get update
    bootstrap_install_missing ca-certificates curl git gnupg xz-utils ffmpeg \
        libglu1-mesa libopengl0 libpulse-mainloop-glib0 libnss3 libsm6 \
        libxcb-cursor0 libxkbcommon-x11-0 libxi6 libxrender1 libxtst6 \
        qt5dxcb-plugin libasound2 libgtk-3-0 dbus-user-session

    install -d -o root -g root -m 0755 /etc/apt/keyrings
    bootstrap_fetch https://download.docker.com/linux/ubuntu/gpg "$staging/docker.asc"
    bootstrap_publish "$staging/docker.asc" /etc/apt/keyrings/docker.asc
    printf '%s\n' 'Types: deb' 'URIs: https://download.docker.com/linux/ubuntu' \
        'Suites: jammy' 'Components: stable' 'Architectures: amd64' \
        'Signed-By: /etc/apt/keyrings/docker.asc' > "$staging/docker.sources"
    bootstrap_publish "$staging/docker.sources" /etc/apt/sources.list.d/docker.sources

    bootstrap_fetch https://nvidia.github.io/libnvidia-container/gpgkey "$staging/nvidia.asc"
    gpg --batch --dearmor --output "$staging/nvidia.gpg" "$staging/nvidia.asc"
    bootstrap_publish "$staging/nvidia.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    bootstrap_fetch https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list "$staging/nvidia.list"
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        "$staging/nvidia.list" > "$staging/nvidia-container-toolkit.list"
    bootstrap_publish "$staging/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    bootstrap_install_missing docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
        docker-compose-plugin nvidia-container-toolkit
    systemctl enable --now docker
    running="$(docker --host unix:///var/run/docker.sock ps -q)"
    [[ -z "$running" ]] || bootstrap_fail 'Containers started during setup; GPU configuration/restart deferred.'
    if [[ -e /etc/docker/daemon.json ]]; then
        cp -a /etc/docker/daemon.json "$staging/daemon.json.before"
    fi
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    docker --host unix:///var/run/docker.sock run --rm --network none --gpus all ubuntu:24.04 nvidia-smi
    usermod -aG docker demo
    loginctl enable-linger demo
    runuser -u demo -- docker --host unix:///var/run/docker.sock info --format '{{.ServerVersion}}'
    printf '\nADMIN STEP COMPLETE — reconnect demo to use Docker; no further sudo is normally needed.\n'
}

bootstrap_main() {
    case "$*" in
        ''|--help|--plan) bootstrap_plan ;;
        '--apply --allow-docker-root-access') bootstrap_apply ;;
        *) bootstrap_fail 'Use --plan, or --apply --allow-docker-root-access after review.' ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then bootstrap_main "$@"; fi
