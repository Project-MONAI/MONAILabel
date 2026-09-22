"""Build the pinned QuPath sources using an isolated, native JDK container."""

import os
import secrets
import shutil
import subprocess
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from monailabel.core.errors import DomainError

# Eclipse Temurin 25 JDK, Ubuntu Noble; includes Linux ARM64.
BUILDER = "eclipse-temurin@sha256:2feab631bffce6236d8bb5261a4abe19a8d6f85bad1c01166f74686c983d011f"


def build_qupath(source: Path, cache: Path, progress: Callable[[str], None]) -> Path:
    if not shutil.which("docker"):
        raise DomainError("QuPath ARM64 setup requires Docker to build its native application.")
    roots = list(source.glob("*/gradlew"))
    if len(roots) != 1:
        raise DomainError("The pinned QuPath source archive is missing its build entry point.")
    directory = roots[0].parent
    log = cache / "qupath-build.log"
    progress("Building native QuPath; the first setup downloads its JDK and dependencies.")
    container = "monailabel-qupath-build-" + secrets.token_hex(8)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "GRADLE_USER_HOME=/src/.gradle-home",
        "--volume",
        f"{directory.resolve()}:/src",
        "--workdir",
        "/src",
        BUILDER,
        "./gradlew",
        "--no-daemon",
        "--max-workers=4",
        "jpackage",
    ]
    try:
        with log.open("w") as output:
            subprocess.run(
                command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800
            )
    except (OSError, subprocess.SubprocessError) as error:
        # Killing a timed-out Docker client does not stop its build container.
        # Remove only this invocation's disposable builder, never another runtime.
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["docker", "rm", "--force", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        raise DomainError(f"Native QuPath build failed. See {log}.") from error
    return directory / "build/dist"
