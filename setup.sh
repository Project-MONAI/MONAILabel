#!/usr/bin/env bash
# Install missing prerequisites and prepare the existing, pinned viewer recipes.
set -euo pipefail

SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_TEMP=""
SETUP_PENDING=()
SETUP_DOCKER=(docker)
SETUP_APT_UPDATED=0
SETUP_NODE_ARCH=x64

usage() {
    cat <<'EOF'
Usage: ./setup.sh [--check] [--cpu]

Prepare MONAI Label on Ubuntu 22.04+ or Debian 12+.
Installs missing system libraries, uv/Python, Node/Corepack, Docker and
NVIDIA Container Toolkit, then prepares Slicer, QuPath, OHIF and CVAT images.
Existing installations and cached downloads are reused.
ARM64 setup prepares the server and OHIF experimentally; managed Slicer,
QuPath and CVAT currently require x86_64. See docs/spark.md.

  --check   Check prerequisites without installing or downloading anything.
  --cpu     Skip Docker/GPU setup; use a hosted chat model and CPU-capable models.
  --help    Show this help.

Run as your normal user; sudo is used only for system packages/configuration.
A working NVIDIA driver and a browser are OS prerequisites. Setup never
changes GPU drivers, reboots, or restarts Docker while containers are running.
Docker group access may require signing out and back in.
MONAILABEL_DATA_DIR, MONAILABEL_CACHE_DIR and MONAILABEL_TOOLS_DIR are respected.
EOF
}

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }
pending() { SETUP_PENDING+=("$*"); printf 'Needs attention: %s\n' "$*" >&2; }
has() { command -v "$1" >/dev/null 2>&1; }

as_root() {
    has sudo || die "Install sudo or ask your administrator to install the missing system packages."
    sudo "$@"
}

installed() {
    [[ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null)" == installed ]]
}

apt_install() {
    local package missing=()
    for package in "$@"; do
        if ! installed "$package"; then missing+=("$package"); fi
    done
    ((${#missing[@]})) || return 0
    say "Installing ${missing[*]}"
    if (( ! SETUP_APT_UPDATED )); then
        as_root apt-get update
        SETUP_APT_UPDATED=1
    fi
    # Never remove an existing package to make an installation fit.
    as_root apt-get install -y --no-remove --no-install-recommends "${missing[@]}"
}

platform_setup() {
    [[ "$(uname -s)" == Linux ]] || die "Automatic setup requires Linux."
    case "$(uname -m)" in
        x86_64) SETUP_NODE_ARCH=x64 ;;
        aarch64|arm64) SETUP_NODE_ARCH=arm64 ;;
        *) die "Automatic setup supports x86_64 and ARM64. See README.md." ;;
    esac
    # shellcheck disable=SC1091
    source /etc/os-release
    case "$ID" in
        ubuntu) (( ${VERSION_ID%%.*} >= 22 )) || die "Ubuntu 22.04+ is required." ;;
        debian) (( ${VERSION_ID%%.*} >= 12 )) || die "Debian 12+ is required." ;;
        *) die "Automatic setup supports Ubuntu and Debian. See README.md for manual prerequisites." ;;
    esac
    SETUP_PACKAGES=(ca-certificates curl git gnupg xz-utils ffmpeg libglu1-mesa
        libopengl0 libpulse-mainloop-glib0 libnss3 libsm6 libxcb-cursor0
        libxkbcommon-x11-0 libxi6 libxrender1 libxtst6 qt5dxcb-plugin)
    if { [[ "$ID" == ubuntu ]] && (( ${VERSION_ID%%.*} >= 24 )); } ||
       { [[ "$ID" == debian ]] && (( ${VERSION_ID%%.*} >= 13 )); }; then
        SETUP_PACKAGES+=(libasound2t64 libgtk-3-0t64)
    else
        SETUP_PACKAGES+=(libasound2 libgtk-3-0)
    fi
}

fetch() {
    local url="$1" target="$2"
    if ! curl --fail --location --show-error --silent --retry 3 \
        --connect-timeout 30 --max-time 900 --proto '=https' --proto-redir '=https' \
        "$url" -o "$target.part"; then
        rm -f -- "$target.part"
        return 1
    fi
    mv -- "$target.part" "$target"
}

ensure_python() {
    if ! has uv; then
        if [[ -x "$HOME/.local/bin/uv" ]]; then
            export PATH="$HOME/.local/bin:$PATH"
        else
            say "Installing uv"
            fetch https://astral.sh/uv/0.11.8/install.sh "$SETUP_TEMP/uv-install.sh"
            UV_INSTALL_DIR="$HOME/.local/bin" sh "$SETUP_TEMP/uv-install.sh"
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
    say "Preparing Python and project dependencies"
    env -u VIRTUAL_ENV uv sync --locked
    [[ -x "$SETUP_ENV/bin/python" ]] || die "Python environment was not created at $SETUP_ENV."
    export PATH="$SETUP_ENV/bin:$PATH"
    # Only configure filesystem defaults; never open a database or start a server.
    MONAILABEL_TOOLS_DIR="$("$SETUP_ENV/bin/python" -c \
        'import os; from monailabel.server.workspace import configure_workspace; configure_workspace(None); print(os.environ["MONAILABEL_TOOLS_DIR"])')"
    export MONAILABEL_TOOLS_DIR
    mkdir -p "$MONAILABEL_TOOLS_DIR"
}

node_ready() {
    has node && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' &&
        has corepack && corepack --version >/dev/null 2>&1
}

ensure_node() {
    if node_ready; then say "Reusing Node.js and Corepack"; return; fi
    # Official Node 22 LTS archive includes npm and Corepack. Keep it project-local.
    local version=22.23.2
    local checksum=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
    if [[ "$SETUP_NODE_ARCH" == arm64 ]]; then
        checksum=fff4078c5def658577f92c88db7db3bc0072924bfb93fe52c1e744a54e94abb8
    fi
    local name="node-v${version}-linux-${SETUP_NODE_ARCH}" archive destination executable
    archive="$MONAILABEL_TOOLS_DIR/downloads/$name.tar.xz"
    destination="$MONAILABEL_TOOLS_DIR/node/$name"
    if [[ ! -x "$destination/bin/node" ]]; then
        mkdir -p "$(dirname "$archive")" "$(dirname "$destination")"
        if [[ ! -f "$archive" ]]; then
            say "Downloading Node.js $version"
            fetch "https://nodejs.org/dist/v$version/$name.tar.xz" "$archive"
        fi
        if ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status; then
            die "Node archive checksum failed: $archive. Remove that file and rerun setup."
        fi
        tar -xJf "$archive" -C "$SETUP_TEMP" --no-same-owner
        [[ ! -e "$destination" ]] || die "Incomplete Node installation at $destination; move it aside and retry."
        mv "$SETUP_TEMP/$name" "$destination"
    fi
    for executable in node npm npx corepack; do
        if [[ -e "$SETUP_ENV/bin/$executable" && ! -L "$SETUP_ENV/bin/$executable" ]]; then
            die "Refusing to replace $SETUP_ENV/bin/$executable."
        fi
        ln -sfn "$destination/bin/$executable" "$SETUP_ENV/bin/$executable"
    done
    hash -r
    node_ready || die "Node.js/Corepack setup failed."
}

ensure_docker() {
    if ! has docker; then
        say "Installing Docker from its official package repository"
        fetch "https://download.docker.com/linux/$ID/gpg" "$SETUP_TEMP/docker.asc"
        as_root install -D -m 644 "$SETUP_TEMP/docker.asc" /etc/apt/keyrings/docker.asc
        cat > "$SETUP_TEMP/docker.sources" <<EOF
Types: deb
URIs: https://download.docker.com/linux/$ID
Suites: $VERSION_CODENAME
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
        as_root install -m 644 "$SETUP_TEMP/docker.sources" /etc/apt/sources.list.d/docker.sources
        SETUP_APT_UPDATED=0
        apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi
    local endpoint
    endpoint="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
    if [[ "$endpoint" != unix:///var/run/docker.sock && "$endpoint" != unix:///run/docker.sock ]]; then
        die "Setup needs the local Docker Engine context; current endpoint is $endpoint. Configure that runtime separately."
    fi
    if ! docker info >/dev/null 2>&1; then
        if ! as_root docker --host "$endpoint" info >/dev/null 2>&1; then
            as_root systemctl start docker
        fi
        as_root docker --host "$endpoint" info >/dev/null || die "Docker Engine is unavailable."
        SETUP_DOCKER=(sudo docker --host "$endpoint")
        as_root usermod -aG docker "$(id -un)"
        pending "Sign out and back in to activate Docker group access, then rerun ./setup.sh."
    fi
}

ensure_container_gpu() {
    local runtime running
    runtime="$("${SETUP_DOCKER[@]}" info --format '{{if index .Runtimes "nvidia"}}ready{{end}}')" ||
        die "Cannot inspect Docker runtimes."
    if [[ "$runtime" == ready ]]; then
        say "Reusing NVIDIA container runtime"
        return
    fi
    if ! has nvidia-ctk; then
        say "Installing NVIDIA Container Toolkit"
        fetch https://nvidia.github.io/libnvidia-container/gpgkey "$SETUP_TEMP/nvidia.asc"
        gpg --batch --yes --dearmor --output "$SETUP_TEMP/nvidia.gpg" "$SETUP_TEMP/nvidia.asc"
        as_root install -D -m 644 "$SETUP_TEMP/nvidia.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        fetch https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list "$SETUP_TEMP/nvidia.list"
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            "$SETUP_TEMP/nvidia.list" > "$SETUP_TEMP/nvidia-container-toolkit.list"
        as_root install -m 644 "$SETUP_TEMP/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
        SETUP_APT_UPDATED=0
        apt_install nvidia-container-toolkit
    fi
    running="$("${SETUP_DOCKER[@]}" ps -q)" || die "Cannot inspect running containers; Docker was not changed."
    as_root nvidia-ctk runtime configure --runtime=docker
    if [[ -n "$running" ]]; then
        pending "NVIDIA runtime configured. When existing containers can be stopped, run sudo systemctl restart docker and rerun ./setup.sh."
    else
        as_root systemctl restart docker
    fi
}

docker_gpu_ready() {
    [[ "$(docker info --format '{{if index .Runtimes "nvidia"}}ready{{end}}')" == ready ]]
}

probe() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then printf 'OK       %s\n' "$label";
    else pending "$label"; fi
}

check_prerequisites() {
    local package
    for package in "${SETUP_PACKAGES[@]}"; do probe "$package" installed "$package"; done
    probe uv has uv
    probe 'Python environment (run ./setup.sh)' "$SETUP_ENV/bin/python" -B -c \
        'import monailabel.server, monai, sam2; import sys; assert sys.version_info >= (3, 12)'
    probe 'Node.js 22+ and Corepack' node_ready
    if (( ! SETUP_CPU )); then
        probe 'NVIDIA GPU driver (nvidia-smi)' nvidia-smi -L
        probe 'Docker access for this user' docker info
        probe 'NVIDIA container runtime' docker_gpu_ready
    fi
}

main() {
    local check=0 arg viewer original_path="$PATH" viewers=(slicer qupath ohif)
    SETUP_CPU=0
    for arg in "$@"; do
        case "$arg" in
            --help|-h) usage; return ;;
            --check) check=1 ;;
            --cpu) SETUP_CPU=1 ;;
            *) usage >&2; die "Unknown option: $arg" ;;
        esac
    done
    cd "$SETUP_ROOT"
    platform_setup
    if [[ "$SETUP_NODE_ARCH" == arm64 ]]; then
        viewers=(ohif)
        say "Experimental ARM64 server setup. Managed Slicer, QuPath and CVAT require x86_64; see docs/spark.md."
    fi
    SETUP_ENV="${UV_PROJECT_ENVIRONMENT:-$SETUP_ROOT/.venv}"
    [[ "$SETUP_ENV" == /* ]] || SETUP_ENV="$SETUP_ROOT/$SETUP_ENV"
    export PATH="$SETUP_ENV/bin:$PATH"
    if (( check )); then
        check_prerequisites
    else
        [[ "$(id -u)" != 0 ]] || die "Run ./setup.sh as your normal user, without sudo."
        SETUP_TEMP="$(mktemp -d)"
        trap 'rm -rf -- "$SETUP_TEMP"' EXIT
        apt_install "${SETUP_PACKAGES[@]}"
        ensure_python
        ensure_node
        if (( ! SETUP_CPU )); then
            if ! nvidia-smi -L >/dev/null 2>&1; then
                pending "Install a compatible NVIDIA driver using your OS driver manager, reboot if required, and rerun ./setup.sh. Use --cpu to skip local GPU/chat setup."
            fi
            ensure_docker
            ensure_container_gpu
            if ((${#SETUP_PENDING[@]} == 0)); then
                say "Checking GPU access in Docker"
                "${SETUP_DOCKER[@]}" run --rm --gpus all ubuntu:24.04 nvidia-smi
            fi
        fi
        for viewer in "${viewers[@]}"; do
            say "Preparing $viewer"
            env -u VIRTUAL_ENV uv run --locked --no-sync monailabel viewer "$viewer"
        done
        if [[ "$SETUP_NODE_ARCH" == x64 ]] && docker info >/dev/null 2>&1; then
            say "Preparing CVAT images"
            env -u VIRTUAL_ENV uv run --locked --no-sync monailabel viewer cvat
        fi
        if ! PATH="$original_path" command -v uv >/dev/null 2>&1; then
            pending "Open a new terminal or run: source \"$HOME/.local/bin/env\""
        fi
    fi
    if ((${#SETUP_PENDING[@]})); then
        printf '\nSetup needs attention:\n'
        printf '  - %s\n' "${SETUP_PENDING[@]}"
        return 1
    fi
    if (( check )); then say "Prerequisites are ready";
    elif (( SETUP_CPU )); then say "Ready. Start uv run monailabel-server with your hosted chat settings (docs/coordinator.md).";
    elif [[ "$SETUP_NODE_ARCH" == arm64 ]]; then
        say "ARM64 dependencies prepared. Next run the GPU smoke check in docs/spark.md before starting the server."
    else say "Ready. Start with: uv run monailabel-server"; fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
