"""Provision browser desktop assets and isolated native viewer containers."""

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import httpx
from filelock import FileLock

from monailabel.core.errors import DomainError
from monailabel.viewers.manager import ViewerManager
from monailabel.viewers.provisioning.desktop import extract

RESOURCES = Path(str(files("monailabel.viewers").joinpath("resources")))
NOVNC_VERSION = "1.6.0"
NOVNC_SHA256 = "5066103959ef4e9b10f37e5a148627360dd8414e4cf8a7db92bdbd022e728aaa"
logger = logging.getLogger(__name__)


class BrowserDesktop:
    def __init__(self, root: Path, namespace: str, tools: Path | None = None):
        self.root = root.resolve()
        self.namespace = namespace
        self.viewers = ViewerManager(tools)
        self.dist = self.viewers.root / "desktop" / f"novnc-{NOVNC_VERSION}"
        self.sockets = Path(tempfile.gettempdir()) / ("monailabel-desktop-" + namespace)

    @staticmethod
    def run(*args: str, timeout: int = 60) -> str:
        try:
            result = subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DomainError(
                "Browser desktop requires a running Docker service on Linux.", status=503
            ) from exc
        if result.returncode:
            # Docker logs can include private session paths and application output.
            logger.error("Browser desktop Docker %s failed: %s", args[0], result.stderr[-4000:])
            raise DomainError(
                "Browser desktop setup failed. Check Docker and the server logs.", status=503
            )
        return result.stdout.strip()

    def prepare(self, progress: Callable[[str], None]) -> str:
        if (
            platform.system() != "Linux"
            or platform.machine().casefold() not in {"x86_64", "aarch64", "arm64"}
            or not shutil.which("docker")
        ):
            raise DomainError(
                "Browser desktop requires Docker on a Linux x86_64 or ARM64 server.", status=503
            )
        context = RESOURCES / "desktop"
        digest = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(context.iterdir()) if p.is_file())
        )
        image = "monailabel-desktop:" + digest.hexdigest()[:20]
        cache = self.viewers.root / "desktop"
        cache.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache / "setup.lock"), timeout=900):
            if not self.run("image", "ls", "-q", image):
                progress("Preparing browser desktop; the first launch downloads its runtime.")
                self.run("build", "--tag", image, str(context), timeout=900)
            if not (self.dist / "vnc.html").is_file():
                progress("Preparing the browser display client.")
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    archive = Path(temporary) / "novnc.tar.gz"
                    url = (
                        f"https://codeload.github.com/novnc/noVNC/tar.gz/refs/tags/v{NOVNC_VERSION}"
                    )
                    response = httpx.get(url, follow_redirects=True, timeout=60)
                    response.raise_for_status()
                    if hashlib.sha256(response.content).hexdigest() != NOVNC_SHA256:
                        raise DomainError("The browser display client checksum did not match.")
                    archive.write_bytes(response.content)
                    extract(archive, Path(temporary) / "unpacked", "tar.gz")
                    (Path(temporary) / "unpacked" / f"noVNC-{NOVNC_VERSION}").rename(self.dist)
        return image

    def name(self, identifier: str) -> str:
        return f"monailabel-desktop-{self.namespace}-{identifier}"

    def socket(self, identifier: str) -> Path:
        return self.sockets / identifier / "rfb.sock"

    def running(self, identifier: str) -> bool:
        return bool(self.run("ps", "-q", "--filter", f"name=^/{self.name(identifier)}$"))

    def start(
        self,
        identifier: str,
        viewer: str,
        configuration: dict[str, object],
        progress: Callable[[str], None],
    ) -> None:
        image = self.prepare(progress)
        installation = self.viewers.ensure(viewer, progress=progress)
        executable = Path(installation.executable).resolve()
        installation_root = executable.parent if viewer == "slicer" else executable.parent.parent
        directory = self.root / identifier
        directory.mkdir(parents=True, mode=0o700)
        os.chmod(directory, 0o700)
        self.sockets.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.socket(identifier).parent.mkdir(mode=0o700)
        for filename, value in {
            "launch.json": configuration,
            "runtime.json": {
                "viewer": viewer,
                "executable": "/viewer/" + executable.relative_to(installation_root).as_posix(),
            },
        }.items():
            with (directory / filename).open("x") as stream:
                os.chmod(stream.fileno(), 0o600)
                json.dump(value, stream)
        progress("Starting the private viewer desktop.")
        self.run(
            "run",
            "-d",
            "--name",
            self.name(identifier),
            "--init",
            "--network",
            "host",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=1g",
            "--shm-size",
            "1g",
            "--pids-limit",
            "512",
            "--memory",
            os.environ.get("MONAILABEL_DESKTOP_MEMORY", "8g"),
            "--cpus",
            os.environ.get("MONAILABEL_DESKTOP_CPUS", "4"),
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=2",
            "--volume",
            f"{directory}:/session:rw",
            "--volume",
            f"{self.socket(identifier).parent}:/connection:rw",
            "--volume",
            f"{installation_root}:/viewer:ro",
            "--volume",
            f"{RESOURCES}:/bridges:ro",
            image,
        )
        for _ in range(180):
            progress("Waiting for the native viewer to become ready.")
            if not self.running(identifier):
                raise DomainError(
                    "The desktop could not start. Check its container log and retry.", status=503
                )
            if self.socket(identifier).exists() and (directory / "started").exists():
                return
            time.sleep(1)
        raise DomainError(
            "The viewer desktop did not become ready within three minutes.", status=503
        )

    def stop(self, identifier: str) -> None:
        if self.run("ps", "-aq", "--filter", f"name=^/{self.name(identifier)}$"):
            self.save_log(identifier)
            self.run("rm", "-f", self.name(identifier))
        (self.root / identifier / "launch.json").unlink(missing_ok=True)
        shutil.rmtree(self.socket(identifier).parent, ignore_errors=True)

    def save_log(self, identifier: str) -> None:
        try:
            output = subprocess.run(
                ["docker", "logs", "--tail", "200", self.name(identifier)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # The native viewer can write to its session directory. Replace the log
            # atomically so a viewer-created symlink cannot redirect a host write.
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                log = Path(temporary) / "runtime.log"
                log.write_text(output.stdout + output.stderr)
                log.chmod(0o600)
                log.replace(self.root / identifier / "runtime.log")
        except (OSError, subprocess.SubprocessError):
            logger.warning("Could not save the browser desktop log", exc_info=True)
