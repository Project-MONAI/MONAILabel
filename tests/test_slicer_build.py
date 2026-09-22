"""Safety checks for the opt-in, resumable native Slicer source recipe."""

import runpy
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_existing_slicer_build_exports_one_quoted_executable(tmp_path, capsys):
    recipe = runpy.run_path(str(ROOT / "examples/build_slicer_arm64.py"))
    executable = tmp_path / "Slicer user's build $value" / "Slicer"
    executable.parent.mkdir()
    executable.touch(mode=0o700)
    executable.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\xb7\x00")
    recipe["show_installation"](tmp_path)
    command = shlex.split(capsys.readouterr().out)
    assert command == ["export", f"MONAILABEL_SLICER_EXECUTABLE={executable}"]


def test_incomplete_slicer_installation_is_not_overwritten(tmp_path):
    recipe = runpy.run_path(str(ROOT / "examples/build_slicer_arm64.py"))
    with pytest.raises(RuntimeError, match="Incomplete portable installation"):
        recipe["show_installation"](tmp_path)


def test_slicer_installation_rejects_an_x86_launcher(tmp_path):
    recipe = runpy.run_path(str(ROOT / "examples/build_slicer_arm64.py"))
    executable = tmp_path / "Slicer-build" / "Slicer"
    executable.parent.mkdir()
    executable.touch(mode=0o700)
    executable.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\x3e\x00")
    with pytest.raises(RuntimeError, match="not a native ARM64"):
        recipe["show_installation"](tmp_path)


def test_slicer_source_rejects_a_different_revision(tmp_path, monkeypatch):
    recipe = runpy.run_path(str(ROOT / "examples/build_slicer_arm64.py"))
    (tmp_path / "source").mkdir()
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "other-revision\n")
    with pytest.raises(RuntimeError, match="Unexpected source revision"):
        recipe["source"](tmp_path, "source", "https://example.invalid/repo", "pinned-revision")


def test_unchanged_recipe_patch_preserves_build_dependency_timestamp(tmp_path):
    recipe = runpy.run_path(str(ROOT / "examples/build_slicer_arm64.py"))
    source = tmp_path / "CMakeLists.txt"
    recipe["write_changed"](source, "original")
    original = source.stat().st_mtime_ns
    recipe["write_changed"](source, "original")
    assert source.stat().st_mtime_ns == original
    recipe["write_changed"](source, "changed")
    assert source.read_text() == "changed"
