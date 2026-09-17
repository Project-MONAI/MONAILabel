"""Thin desktop adapter: request provisioning, then launch an authenticated viewer session."""

import json
import os
import subprocess
from collections.abc import Callable, Collection
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from platformdirs import user_cache_path, user_data_path

from monailabel.core.errors import DomainError
from monailabel.viewers.provisioning.catalog import Installation, ToolSpec, release
from monailabel.viewers.provisioning.desktop import DesktopProvisioner

# Kept as convenient explicit profiles for callers/tests; release data lives in JSON.
SLICER_LINUX = release("slicer", "Linux", "x86_64")
QUPATH_LINUX = release("qupath", "Linux", "x86_64")


class ViewerManager:
    def __init__(self, root: Path | None = None, spec: ToolSpec | None = None):
        self.root = root or Path(
            os.environ.get("MONAILABEL_TOOLS_DIR", str(user_cache_path("monailabel") / "tools"))
        )
        self.spec = spec
        self.provisioner = DesktopProvisioner(self.root)

    def ensure(
        self,
        viewer: str = "slicer",
        *,
        download: bool = True,
        progress: Callable[[str], None] = print,
    ) -> Installation:
        selected = viewer.casefold().replace(" ", "")
        selected = {"3dslicer": "slicer", "3d-slicer": "slicer"}.get(selected, selected)
        if selected not in {"slicer", "qupath"}:
            raise DomainError("Choose 3D Slicer or QuPath for a desktop launch.")
        return self.provisioner.ensure(
            selected, spec=self.spec, download=download, progress=progress
        )

    def launch(
        self,
        installation: Installation,
        url: str,
        project_id: str,
        *,
        asset_id: str | None = None,
        mode: str = "annotation",
        shared_filesystem: bool = False,
        token: str | None = None,
        secret_env: Collection[str] = (),
    ) -> dict[str, str | int]:
        if installation.viewer == "qupath":
            from monailabel.viewers.qupath import prepare_extension

            prepare_extension(installation, self.root)
        sessions = self.root / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        session_id = uuid4().hex
        config = sessions / f"{session_id}.json"
        with config.open("x") as stream:
            os.chmod(config, 0o600)
            json.dump(
                {
                    "url": url.rstrip("/"),
                    "project_id": project_id,
                    "asset_id": asset_id,
                    "mode": mode,
                    "shared_filesystem": shared_filesystem,
                    "token": token,
                },
                stream,
            )
        bridge = str(files("monailabel.viewers").joinpath("resources", "slicer_bridge.py"))
        log = sessions / f"{session_id}.log"
        # The viewer needs desktop settings and its own session, not provider credentials.
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in secret_env
            and not key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
        }
        env["MONAILABEL_VIEWER_SESSION"] = str(config)
        env["MONAILABEL_QUPATH_DATA"] = os.environ.get(
            "MONAILABEL_QUPATH_DATA", str(user_data_path("monailabel") / "qupath" / "projects")
        )
        command = (
            [installation.executable, "--quiet"]
            if installation.viewer == "qupath"
            else [installation.executable, "--no-splash", "--python-script", bridge]
        )
        with log.open("wb") as output:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {"pid": process.pid, "log": str(log), "executable": installation.executable}
