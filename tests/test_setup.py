"""Exercise installer failure/reuse paths without changing the host or using the network."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "setup.sh"


def run_setup(tmp_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f"""
source {shlex.quote(str(SETUP))}
SETUP_ROOT="$SETUP_TEST_ROOT"
SETUP_ENV="$SETUP_TEST_ROOT/.venv"
MONAILABEL_TOOLS_DIR="$SETUP_TEST_ROOT/workspace/.cache/tools"
SETUP_TEMP="$SETUP_TEST_ROOT/staging"
as_root() {{ printf '%s\\n' "$*" >> "$SETUP_TEST_ROOT/root-commands"; }}
fetch() {{ echo 'Unexpected download' >&2; exit 90; }}
{script}
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "SETUP_TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_setup_check_is_read_only_and_reports_all_missing_dependencies(tmp_path):
    result = run_setup(
        tmp_path,
        """
platform_setup() { SETUP_PACKAGES=(git curl); }
installed() { [[ "$1" == git ]]; }
node_ready() { return 1; }
nvidia-smi() { return 1; }
docker() { return 1; }
apt_install() { echo 'Unexpected installation'; exit 91; }
main --check
""",
    )
    assert result.returncode == 1
    assert "curl" in result.stderr
    assert "Python environment" in result.stderr
    assert "Node.js" in result.stderr
    assert "NVIDIA GPU driver" in result.stderr
    assert "Docker access" in result.stderr
    assert not list(tmp_path.iterdir())


def test_setup_installs_only_missing_packages_and_refreshes_apt_once(tmp_path):
    result = run_setup(
        tmp_path,
        """
installed() { [[ "$1" == git ]]; }
apt_install git curl
apt_install git
apt_install gnupg
""",
    )
    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "root-commands").read_text().splitlines()
    assert commands == [
        "apt-get update",
        "apt-get install -y --no-remove --no-install-recommends curl",
        "apt-get install -y --no-remove --no-install-recommends gnupg",
    ]


@pytest.mark.parametrize("running", ["active-container", ""])
def test_setup_only_restarts_docker_when_no_containers_are_running(tmp_path, running):
    result = run_setup(
        tmp_path,
        f"""
nvidia-ctk() {{ :; }}
docker() {{ if [[ "$1" == ps ]]; then printf '%s' {shlex.quote(running)}; fi; }}
ensure_container_gpu
printf '%s\\n' "${{SETUP_PENDING[@]}}"
""",
    )
    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "root-commands").read_text()
    assert "nvidia-ctk runtime configure --runtime=docker" in commands
    assert ("systemctl restart docker" in commands) == (not running)
    assert ("When existing containers can be stopped" in result.stdout) == bool(running)


@pytest.mark.parametrize("failure", ["info", "ps"])
def test_setup_does_not_reconfigure_or_restart_an_uninspectable_docker(tmp_path, failure):
    result = run_setup(
        tmp_path,
        f"""
nvidia-ctk() {{ :; }}
docker() {{ [[ "$1" != {shlex.quote(failure)} ]]; }}
ensure_container_gpu
""",
    )
    assert result.returncode == 1
    assert "Cannot inspect" in result.stderr
    assert not list(tmp_path.iterdir())


def test_setup_reuses_configured_docker_without_system_changes(tmp_path):
    result = run_setup(
        tmp_path,
        """
docker() { [[ "$1" == info ]] || exit 92; printf ready; }
ensure_container_gpu
ensure_container_gpu
""",
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_setup_rejects_remote_docker_before_changing_host_configuration(tmp_path):
    result = run_setup(
        tmp_path,
        """
docker() { printf 'ssh://another-machine'; }
unset DOCKER_HOST
ensure_docker
""",
    )
    assert result.returncode == 1
    assert "local Docker Engine context" in result.stderr
    assert not list(tmp_path.iterdir())


def test_setup_rejects_corrupted_node_download_before_extraction(tmp_path):
    downloads = tmp_path / "workspace/.cache/tools/downloads"
    downloads.mkdir(parents=True)
    (downloads / "node-v22.23.2-linux-x64.tar.xz").write_bytes(b"corrupt archive")
    result = run_setup(tmp_path, "node_ready() { return 1; }; ensure_node")
    assert result.returncode == 1
    assert "checksum failed" in result.stderr
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / "staging").exists()


def test_setup_reuses_cached_node_and_exposes_it_to_uv_without_shell_profile_edits(tmp_path):
    cached = tmp_path / "workspace/.cache/tools/node/node-v22.23.2-linux-x64/bin"
    cached.mkdir(parents=True)
    for name in ("node", "npm", "npx", "corepack"):
        executable = cached / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    env_bin = tmp_path / ".venv/bin"
    env_bin.mkdir(parents=True)
    old_bin = tmp_path / "old-node/bin"
    old_bin.mkdir(parents=True)
    (old_bin / "node").write_text("#!/bin/sh\nexit 1\n")
    (old_bin / "node").chmod(0o755)
    result = run_setup(
        tmp_path,
        """
export PATH="$SETUP_ENV/bin:$SETUP_TEST_ROOT/old-node/bin:$PATH"
node --version || true
ensure_node
ensure_node
""",
    )
    assert result.returncode == 0, result.stderr
    for name in ("node", "npm", "npx", "corepack"):
        assert (env_bin / name).resolve() == cached / name
    assert not (tmp_path / "root-commands").exists()


def test_setup_preserves_existing_environment_executables(tmp_path):
    cached = tmp_path / "workspace/.cache/tools/node/node-v22.23.2-linux-x64/bin"
    cached.mkdir(parents=True)
    (cached / "node").write_text("#!/bin/sh\nexit 0\n")
    (cached / "node").chmod(0o755)
    env_bin = tmp_path / ".venv/bin"
    env_bin.mkdir(parents=True)
    (env_bin / "node").write_text("user executable")
    result = run_setup(tmp_path, "node_ready() { return 1; }; ensure_node")
    assert result.returncode == 1
    assert "Refusing to replace" in result.stderr
    assert (env_bin / "node").read_text() == "user executable"
