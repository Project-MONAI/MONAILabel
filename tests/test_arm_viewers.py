"""Native viewer recipes must not silently execute x86 images on ARM64."""

import subprocess

import pytest

from monailabel.core.errors import DomainError
from monailabel.viewers import cvat_runtime
from monailabel.viewers.provisioning import qupath_build
from monailabel.viewers.provisioning.catalog import catalog
from monailabel.viewers.provisioning.desktop import DesktopProvisioner


def test_qupath_recipe_is_pinned_source_on_arm():
    recipes = catalog().releases
    arm = next(
        spec
        for spec in recipes
        if spec.name == "qupath" and spec.machine == "arm64" and spec.system == "Linux"
    )
    assert arm.build == "qupath"
    assert arm.algorithm == "sha256"
    assert len(arm.checksum) == 64


def test_slicer_discovers_the_explicitly_built_native_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("MONAILABEL_SLICER_EXECUTABLE", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("glob.glob", lambda _: [])
    provisioner = DesktopProvisioner(tmp_path)
    assert provisioner.discover("slicer") is None
    executable = tmp_path / "slicer/native-install/Slicer-arm64/Slicer"
    executable.parent.mkdir(parents=True)
    executable.touch(mode=0o700)
    assert provisioner.discover("slicer") == executable
    other = tmp_path / "slicer/native-install/ambiguous/Slicer"
    other.parent.mkdir()
    other.touch(mode=0o700)
    assert provisioner.discover("slicer") is None


def test_qupath_build_uses_native_isolated_jdk_and_bounded_workers(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "gradlew").touch()
    monkeypatch.setattr(qupath_build.shutil, "which", lambda _: "/usr/bin/docker")
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs))
    )
    result = qupath_build.build_qupath(tmp_path, tmp_path, lambda _: None)
    command, options = commands[0]
    assert result == source / "build/dist"
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--max-workers=4" in command
    assert qupath_build.BUILDER in command
    assert "--privileged" not in command
    assert options["check"] and options["timeout"] == 1800


def test_qupath_build_failure_is_actionable(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    (tmp_path / "source/gradlew").touch()
    monkeypatch.setattr(qupath_build.shutil, "which", lambda _: "/usr/bin/docker")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "docker")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(DomainError, match="qupath-build.log"):
        qupath_build.build_qupath(tmp_path, tmp_path, lambda _: None)


@pytest.mark.parametrize("architecture", ["aarch64", "arm64", "x86_64"])
def test_cvat_selects_native_images_only_on_arm(monkeypatch, architecture):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: architecture)
    images = cvat_runtime.image_environment()
    if architecture == "x86_64":
        assert images == {}
    else:
        assert len(images) == 2
        assert all("-arm64-" in name for name in images.values())
        assert cvat_runtime.SOURCE_REVISION[:8] in images["MONAILABEL_CVAT_SERVER_IMAGE"]


def test_cvat_cached_native_images_are_reused_without_rebuilding(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_TOOLS_DIR", str(tmp_path))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "arm64\n")

    monkeypatch.setattr(cvat_runtime.CvatManager, "_run", run)
    cvat_runtime.CvatManager._build_native_images(
        {"MONAILABEL_CVAT_SERVER_IMAGE": "native:test"}, lambda _: None
    )
    assert commands == [
        ["docker", "image", "ls", "-q", "native:test"],
        ["docker", "image", "inspect", "--format", "{{.Architecture}}", "native:test"],
    ]


def test_cvat_rejects_wrong_architecture_under_native_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_TOOLS_DIR", str(tmp_path))
    monkeypatch.setattr(
        cvat_runtime.CvatManager,
        "_run",
        lambda command: subprocess.CompletedProcess(command, 0, "amd64\n"),
    )
    with pytest.raises(DomainError, match="not ARM64"):
        cvat_runtime.CvatManager._build_native_images(
            {"MONAILABEL_CVAT_SERVER_IMAGE": "native:test"}, lambda _: None
        )
