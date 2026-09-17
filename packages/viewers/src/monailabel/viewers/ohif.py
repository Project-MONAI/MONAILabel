"""OHIF runtime adapter: reuse a prepared distribution or invoke the separate Node builder."""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from filelock import FileLock
from platformdirs import user_cache_path

from monailabel.core.errors import DomainError

_RESOURCES = Path(str(files("monailabel.viewers").joinpath("resources")))
_RECIPE = json.loads((_RESOURCES / "installers/ohif.json").read_text())
VERSION: str = _RECIPE["version"]
COMMIT: str = _RECIPE["commit"]


class OhifManager:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(
            os.environ.get("MONAILABEL_TOOLS_DIR", str(user_cache_path("monailabel") / "tools"))
        )
        self.source = self.root / "ohif" / f"source-v{VERSION}"
        self.prepared = os.environ.get("MONAILABEL_OHIF_DIST")
        self.dist = Path(self.prepared) if self.prepared else self.source / "platform/app/dist"

    def ensure(self, progress: Callable[[str], None] = print) -> Path:
        if self.prepared:
            if not (self.dist / "index.html").is_file():
                raise DomainError(
                    "MONAILABEL_OHIF_DIST must contain a prepared MONAI Label OHIF build."
                )
            progress(f"Reusing configured OHIF distribution: {self.dist}")
            return self.dist
        resources = _RESOURCES / "ohif"
        build = _RESOURCES / "installers/ohif-build.mjs"
        digest = hashlib.sha256()
        for path in sorted(resources.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(resources)).encode())
                digest.update(path.read_bytes())
        digest.update(build.read_bytes())
        digest.update((_RESOURCES / "installers/ohif.json").read_bytes())
        signature = digest.hexdigest()
        self.source.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.source.parent / "build.json"
        with FileLock(str(self.source.parent / "build.lock"), timeout=1800):
            if (
                manifest.exists()
                and (self.dist / "index.html").exists()
                and json.loads(manifest.read_text()).get("extension_hash") == signature
            ):
                progress(f"Reusing OHIF {VERSION}")
                return self.dist
            node = shutil.which("node")
            if not node:
                raise DomainError(
                    "OHIF setup needs Node 22+, Git and Corepack, or MONAILABEL_OHIF_DIST."
                )
            log = self.source.parent / "build.log"
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in {"GITHUB_TOKEN", "GH_TOKEN"}
                and not k.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            }
            progress("Preparing OHIF using its pinned Node build recipe")
            with log.open("ab") as output:
                result = subprocess.run(
                    [node, str(build), str(self.source), str(resources)],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            if result.returncode or not (self.dist / "index.html").exists():
                raise DomainError(f"OHIF setup failed. See {log}.")
            manifest.write_text(
                json.dumps(
                    {"version": VERSION, "commit": COMMIT, "extension_hash": signature}, indent=2
                )
                + "\n"
            )
            return self.dist
