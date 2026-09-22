"""Prepare a private CVAT runtime and its pinned browser distribution."""

import hashlib
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import httpx
from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError

VERSION = "2.76.0"
RESOURCES = Path(str(files("monailabel.viewers").joinpath("resources/cvat")))
SOURCE_REVISION = "b78c39f7a5de6450567e77c9e1ffc8f0c428cea7"


def image_environment() -> dict[str, str]:
    """Compose and UI extraction must select the same native images."""
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        return {}
    signature = hashlib.sha256((RESOURCES / "Dockerfile.ui-arm64").read_bytes()).hexdigest()[:12]
    return {
        "MONAILABEL_CVAT_SERVER_IMAGE": (
            f"monailabel-cvat-server:{VERSION}-arm64-{SOURCE_REVISION[:8]}"
        ),
        "MONAILABEL_CVAT_UI_IMAGE": f"monailabel-cvat-ui:{VERSION}-arm64-{signature}",
    }


class CvatManager:
    def __init__(self, root: Path, namespace: str):
        self.root = root / "cvat" / namespace
        self.project = "monailabel-managed-cvat-" + namespace
        self.dist = self.root / f"ui-{VERSION}"

    @staticmethod
    def _run(args: list[str], *, input: str | None = None) -> subprocess.CompletedProcess[str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            and k not in {"MONAILABEL_CVAT_API_PORT", "MONAILABEL_CVAT_UI_PORT"}
        }
        env.update(image_environment())
        result = subprocess.run(
            args, input=input, capture_output=True, text=True, timeout=600, env=env
        )
        if result.returncode:
            # Neither provisioning stdin nor upstream logs belong in an API error.
            raise DomainError(
                "CVAT setup failed. Check that Docker is running and can pull images."
            )
        return result

    @classmethod
    def prepare_images(
        cls, download: bool = True, progress: Callable[[str], None] = print
    ) -> list[str]:
        if not shutil.which("docker"):
            raise DomainError("Install Docker with Compose to prepare CVAT.", status=503)
        command = [
            "docker",
            "compose",
            "-p",
            "monailabel-cvat-download",
            "-f",
            str(RESOURCES / "compose.yaml"),
        ]
        images = sorted(set(cls._run([*command, "config", "--images"]).stdout.split()))
        if download:
            if native := image_environment():
                cls._build_native_images(native, progress)
                cls._run([*command, "pull", "--policy", "missing"])
            else:
                cls._run([*command, "pull"])
        else:
            cls._run(["docker", "image", "inspect", *images])
        return images

    @classmethod
    def _build_native_images(cls, images: dict[str, str], progress: Callable[[str], None]) -> None:
        cache = (
            Path(
                os.environ.get("MONAILABEL_TOOLS_DIR", str(user_cache_path("monailabel") / "tools"))
            )
            / "cvat-builds"
        )
        cache.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache / "build.lock"), timeout=1800):
            for variable, image in images.items():
                if cls._run(["docker", "image", "ls", "-q", image]).stdout.strip():
                    architecture = cls._run(
                        ["docker", "image", "inspect", "--format", "{{.Architecture}}", image]
                    ).stdout.strip()
                    if architecture != "arm64":
                        raise DomainError(
                            f"Cached CVAT image {image} is not ARM64. Remove or retag that "
                            "image before preparing the native viewer."
                        )
                    continue
                progress("Building native ARM64 CVAT; the first setup can take several minutes.")
                command = ["docker", "build", "--platform", "linux/arm64", "--tag", image]
                if variable == "MONAILABEL_CVAT_SERVER_IMAGE":
                    command += [f"https://github.com/cvat-ai/cvat.git#{SOURCE_REVISION}"]
                else:
                    command += ["-f", str(RESOURCES / "Dockerfile.ui-arm64"), str(RESOURCES)]
                log = cache / ("server.log" if variable.endswith("SERVER_IMAGE") else "ui.log")
                try:
                    with log.open("w") as output:
                        subprocess.run(
                            command,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=1800,
                        )
                except (OSError, subprocess.SubprocessError) as error:
                    raise DomainError(f"Native CVAT build failed. See {log}.") from error

    def ensure(self, username: str, password: str, progress: Callable[[str], None]) -> str:
        if not shutil.which("docker"):
            raise DomainError("Install Docker with Compose to prepare the CVAT viewer.", status=503)
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / "setup.lock"), timeout=600):
            configuration = self.root / "runtime.json"
            if configuration.exists():
                port = int(json.loads(configuration.read_text())["port"])
            else:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                configuration.write_text(json.dumps({"port": port, "version": VERSION}))
            environment = self.root / "compose.env"
            environment.write_text(f"MONAILABEL_CVAT_API_PORT={port}\n")
            compose = [
                "docker",
                "compose",
                "--env-file",
                str(environment),
                "-p",
                self.project,
                "-f",
                str(RESOURCES / "compose.yaml"),
            ]
            progress("Preparing CVAT services; the first launch downloads the viewer images.")
            if image_environment():
                self.prepare_images(progress=progress)
            self._run([*compose, "up", "-d", "server", "importer", "chunks", "utilities"])
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 180
            with httpx.Client(base_url=url, timeout=5) as http:
                while time.monotonic() < deadline:
                    try:
                        response = http.get("/api/server/about")
                        if response.status_code == 200:
                            if response.json()["version"] != VERSION:
                                raise DomainError(
                                    "The managed CVAT version does not match its viewer."
                                )
                            break
                    except httpx.TransportError:
                        pass
                    progress("Waiting for CVAT to initialize its database and workers.")
                    time.sleep(1)
                else:
                    raise DomainError(
                        "CVAT did not become ready. Check Docker and retry opening it."
                    )
            code = (
                "from django.contrib.auth import get_user_model\n"
                f"username, password = {username!r}, {password!r}\n"
                "if not get_user_model().objects.filter(username=username).exists():\n"
                "    get_user_model().objects.create_superuser(username, '', password)\n"
            )
            self._run(
                [*compose, "exec", "-T", "server", "python", "manage.py", "shell"], input=code
            )
            if not (self.dist / "index.html").is_file():
                progress("Preparing the CVAT browser interface.")
                ui_image = image_environment().get(
                    "MONAILABEL_CVAT_UI_IMAGE", f"cvat/ui:v{VERSION}"
                )
                if not image_environment():
                    self._run(["docker", "pull", ui_image])
                container = "monailabel-cvat-ui-" + secrets.token_hex(8)
                temporary = self.root / ("ui-" + secrets.token_hex(8))
                self._run(["docker", "create", "--name", container, ui_image])
                try:
                    temporary.mkdir()
                    self._run(
                        ["docker", "cp", f"{container}:/usr/share/nginx/html/.", str(temporary)]
                    )
                    if not (temporary / "index.html").is_file():
                        raise DomainError("The CVAT image did not contain its browser interface.")
                    temporary.rename(self.dist)
                finally:
                    self._run(["docker", "rm", container])
                    shutil.rmtree(temporary, ignore_errors=True)
            return url
