"""Build the thin QuPath extension with QuPath's own bundled Groovy compiler."""

import hashlib
import os
import subprocess
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

from filelock import FileLock

from monailabel.core.errors import DomainError
from monailabel.viewers.provisioning.catalog import Installation


def prepare_extension(installation: Installation, root: Path) -> Path:
    source = Path(str(files("monailabel.viewers").joinpath("resources", "qupath")))
    digest = hashlib.sha256()
    for path in sorted(source.glob("*.groovy")):
        digest.update(path.read_bytes())
    cache = root / "qupath" / "extension" / digest.hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    jar = cache / "monailabel-qupath.jar"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
    }

    def run(script: str, *arguments: str) -> None:
        executable = Path(installation.executable)
        # Windows bundles a console launcher for scripts and captured build output.
        console = executable.with_name(executable.stem + " (console).exe")
        if executable.suffix.lower() == ".exe" and console.is_file():
            executable = console
        command = [str(executable), "script", str(source / script)]
        for argument in arguments:
            command.extend(["--args", argument])
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise DomainError("QuPath extension setup failed: " + result.stdout[-3000:])

    with FileLock(str(cache / "build.lock"), timeout=180):
        if not jar.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                run("build.groovy", str(source), temporary)
                with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in Path(temporary).rglob("*.class"):
                        archive.write(path, str(path.relative_to(temporary)))
                    archive.writestr(
                        "META-INF/services/qupath.lib.gui.extensions.QuPathExtension",
                        "org.monailabel.qupath.MonaiLabelExtension\n",
                    )
        run("install.groovy", str(jar), str(root / "qupath" / "user-data"))
    return jar
