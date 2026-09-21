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
