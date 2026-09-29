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

"""Container cleanup remains reliable even when native log files are untrusted."""

import stat
import subprocess
from types import SimpleNamespace

import pytest

from monailabel.viewers.browser_desktop import BrowserDesktop


@pytest.mark.parametrize("log_failure", [False, True])
def test_stop_preserves_host_files_and_removes_runtime(tmp_path, monkeypatch, log_failure):
    runtime = BrowserDesktop(tmp_path / "sessions", "fixture", tmp_path / "tools")
    runtime.sockets = tmp_path / "sockets"
    directory = runtime.root / "viewer"
    directory.mkdir(parents=True)
    launch = directory / "launch.json"
    launch.write_text("private launch configuration")
    connection = runtime.socket("viewer").parent
    connection.mkdir(parents=True)
    target = tmp_path / "preserved.txt"
    target.write_text("Keep this host file")
    (directory / "runtime.log").symlink_to(target)
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return "container" if args[0] == "ps" else ""

    def logs(*args, **kwargs):
        if log_failure:
            raise subprocess.TimeoutExpired("docker logs", 30)
        return SimpleNamespace(stdout="Viewer output", stderr="")

    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(subprocess, "run", logs)
    runtime.stop("viewer")
    assert calls[-1] == ("rm", "-f", runtime.name("viewer"))
    assert not launch.exists()
    assert not connection.exists()
    assert target.read_text() == "Keep this host file"
    if not log_failure:
        log = directory / "runtime.log"
        assert not log.is_symlink()
        assert log.read_text() == "Viewer output"
        assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_server_container_stages_host_visible_bridges(tmp_path, monkeypatch):
    from monailabel.viewers.browser_desktop import RESOURCES

    monkeypatch.setenv("MONAILABEL_DESKTOP_SOCKET_DIR", str(tmp_path / "socket-mount"))
    runtime = BrowserDesktop(tmp_path / "workspace/sessions", "fixture", tmp_path / "tools")
    assert runtime.sockets.is_relative_to(tmp_path / "socket-mount")
    executable = tmp_path / "tools/slicer/Slicer"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(runtime, "prepare", lambda progress: "fixture-image")
    monkeypatch.setattr(
        runtime.viewers,
        "ensure",
        lambda *args, **kwargs: SimpleNamespace(executable=str(executable)),
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        (runtime.root / "viewer/started").touch()
        runtime.socket("viewer").touch()
        return ""

    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(runtime, "running", lambda identifier: True)
    runtime.start("viewer", "slicer", {}, lambda message: None)
    mount = next(arg for arg in calls[0] if arg.endswith(":/bridges:ro"))
    from pathlib import Path

    bridges = Path(mount.removesuffix(":/bridges:ro"))
    assert bridges.is_relative_to(runtime.viewers.root)
    assert bridges != RESOURCES
    assert (bridges / "slicer_bridge.py").read_bytes() == (
        RESOURCES / "slicer_bridge.py"
    ).read_bytes()
