import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.viewers.manager import SLICER_LINUX, Installation, ViewerManager
from monailabel.viewers.provisioning.desktop import DesktopProvisioner
from monailabel.viewers.resources.geometry import (
    clear_labels,
    merge_proposal,
    roi_geometry,
    slicer_reverses_slices,
    source_slice,
)


@pytest.mark.parametrize("left_handed", [False, True])
@pytest.mark.parametrize("oblique", [False, True])
def test_slicer_slice_order_maps_source_voxels_without_resampling(left_handed, oblique):
    shape = [7, 9, 11]
    affine = np.diag([-1.2 if left_handed else 1.2, 1.5, 2.0, 1])
    affine[:3, 3] = [42, -30, 10]
    if oblique:
        angle = np.deg2rad(27)
        rotation = np.eye(4)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        affine = rotation @ affine
    loaded = affine.copy()
    if left_handed:
        loaded[:, 2] = -affine[:, 2]
        loaded[:, 3] = affine[:, 3] + affine[:, 2] * (shape[2] - 1)
    assert slicer_reverses_slices(affine, loaded, shape) == left_handed
    # The same point in the displayed image and a source-grid mask stays in the same RAS place.
    source = np.array([2, 3, 4, 1])
    display = source.copy()
    if left_handed:
        display[2] = shape[2] - 1 - source[2]
    np.testing.assert_allclose(loaded @ display, affine @ source)


@pytest.mark.parametrize("change", ["shift", "scale", "rotate", "nan"])
def test_slicer_slice_order_rejects_unmapped_geometry(change):
    source = np.eye(4)
    loaded = source.copy()
    if change == "shift":
        loaded[0, 3] = 0.1
    elif change == "scale":
        loaded[2, 2] = 1.1
    elif change == "rotate":
        loaded[:2, :2] = [[0, -1], [1, 0]]
    else:
        loaded[0, 0] = np.nan
    with pytest.raises(ValueError, match="source voxel grid"):
        slicer_reverses_slices(source, loaded, [7, 9, 11])


@pytest.mark.parametrize("axis", [0, 1, 2, None])
@pytest.mark.parametrize("labels", [[1], [1, 2]])
def test_clear_preserves_other_structures_and_outside_slices(axis, labels):
    current = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 3).astype(np.uint8)
    before = current.copy()
    scope = None if axis is None else {"axis": axis, "index": 2}
    result = clear_labels(current, labels, scope)
    for coordinate in np.ndindex(current.shape):
        expected = (
            0
            if current[coordinate] in labels and (axis is None or coordinate[axis] == 2)
            else current[coordinate]
        )
        assert result[coordinate] == expected
    np.testing.assert_array_equal(current, before)


@pytest.mark.parametrize(
    "labels,scope",
    [
        ([], None),
        ([0], None),
        ([256], None),
        ([1], {"axis": 3, "index": 0}),
        ([1], {"axis": 0, "index": -1}),
        ([1], {"axis": 1, "index": 5}),
        ([1], {}),
    ],
)
def test_invalid_clear_never_changes_input(labels, scope):
    current = np.ones((4, 5, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        clear_labels(current, labels, scope)
    assert current.all()


def archive(name="Slicer-test/Slicer"):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        content = b"#!/bin/sh\nexit 0\n"
        info.size = len(content)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def test_system_installation_wins_over_download(tmp_path, monkeypatch):
    executable = tmp_path / "Slicer"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("MONAILABEL_SLICER_EXECUTABLE", str(executable))
    install = ViewerManager(root=tmp_path / "cache").ensure(download=False, progress=lambda _: None)
    assert install.source == "system"
    assert install.executable == str(executable)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("flip", [1, -1])
def test_roi_covers_inclusive_slice_edges_under_rotation_and_spacing(flip):
    from itertools import product

    angle = np.deg2rad(30)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, flip],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.8, 1.2, 3])
    affine[:3, 3] = [-120, 15, 280]
    transform, size = roi_geometry(affine, [[10, 20, 69], [30, 40, 79]])
    assert np.allclose(size, [16.8, 25.2, 33])
    corners = np.array(
        [
            np.linalg.inv(affine) @ transform @ np.append(np.array(sign) * size / 2, 1)
            for sign in product([-1, 1], repeat=3)
        ]
    )
    assert np.allclose(corners[:, :3].min(axis=0), [9.5, 19.5, 68.5])
    assert np.allclose(corners[:, :3].max(axis=0), [30.5, 40.5, 79.5])
    _, single_size = roi_geometry(affine, [[10, 20, 69], [30, 40, 69]])
    assert single_size[2] == 3


def test_roi_rejects_sheared_geometry():
    affine = np.eye(4)
    affine[0, 1] = 0.2
    with pytest.raises(ValueError, match="orthogonal"):
        roi_geometry(affine, [[1, 2, 3], [4, 5, 6]])


def test_download_verified_then_reused(tmp_path, monkeypatch):
    content = archive()
    monkeypatch.setattr(DesktopProvisioner, "discover", lambda self, viewer: None)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    calls = []

    def download(*args, **kw):
        calls.append(args)
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
        ).stream("GET", "https://download.example/slicer")

    monkeypatch.setattr(httpx, "stream", download)
    spec = replace(SLICER_LINUX, checksum=hashlib.sha512(content).hexdigest())
    manager = ViewerManager(root=tmp_path, spec=spec)
    first = manager.ensure(progress=lambda _: None)
    second = manager.ensure(progress=lambda _: None)
    assert first.source == "download" and second.source == "cache"
    assert first.executable == second.executable
    assert len(calls) == 1


@pytest.mark.parametrize("wrong_hash", [True, False])
def test_unsafe_download_never_installs(tmp_path, monkeypatch, wrong_hash):
    content = archive("../escape")
    monkeypatch.setattr(DesktopProvisioner, "discover", lambda self, viewer: None)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **kw: httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
        ).stream("GET", "https://download.example/slicer"),
    )
    spec = replace(
        SLICER_LINUX, checksum="wrong" if wrong_hash else hashlib.sha512(content).hexdigest()
    )
    with pytest.raises(DomainError, match="checksum|unsafe path"):
        ViewerManager(root=tmp_path, spec=spec).ensure(progress=lambda _: None)
    assert not (tmp_path.parent / "escape").exists()
    assert not list(tmp_path.rglob("installation.json"))


def test_slice_mapping_and_preservation_of_unsaved_edits():
    shape = [5, 6, 7]
    transform = np.eye(4)
    transform[2, 3] = 3
    scope = source_slice(np.eye(4), transform, shape, [0, 1])
    assert scope == {
        "axis": 2,
        "index": 3,
        "window": [0, 1],
        "orientation": {"transpose": True, "flip_rows": True, "flip_columns": False},
    }
    current = np.zeros(shape, dtype=np.uint8)
    current[1, 1, 1] = 2
    current[2, 2, 3] = 2
    proposed = np.zeros(shape, dtype=np.uint8)
    proposed[3, 3, 3] = 1
    result = merge_proposal(current, proposed, [1], scope)
    assert result[1, 1, 1] == 2 and result[2, 2, 3] == 2 and result[3, 3, 3] == 1
    proposed[2, 2, 3] = 1
    with pytest.raises(ValueError, match="overlaps"):
        merge_proposal(current, proposed, [1], scope)
    oblique = np.eye(4)
    oblique[2, 0] = 0.3
    with pytest.raises(ValueError, match="Align"):
        source_slice(np.eye(4), oblique, shape, [0, 1])


@pytest.mark.parametrize("mode", ["annotation", "review"])
def test_browser_launch_uses_current_user_and_sample(
    client, http, seeded, monkeypatch, tmp_path, mode
):
    setup, assets = seeded
    if mode == "review":
        client.post(
            f"/api/assets/{assets[0]['id']}/review",
            {
                "base_revision": 0,
                "mask": client.get(f"/api/assets/{assets[0]['id']}/fixture")["mask"],
                "covered_labels": [0, 1, 2],
            },
        )
    client.post(
        f"/api/projects/{setup['project_id']}/models",
        {
            "name": "Remote",
            "provider": "http-mask",
            "label_ids": [0, 1],
            "config": {"url": "https://provider.example/predict", "token_env": "CUSTOM_AUTH"},
        },
    )
    calls = []
    monkeypatch.setattr(ViewerManager, "ensure", lambda self, *args, **kw: object())
    http.headers["host"] = "localhost"

    expected_mode = mode

    def launch(
        self, installation, url, project_id, *, asset_id, token, secret_env, mode, shared_filesystem
    ):
        assert mode == expected_mode
        assert shared_filesystem
        assert "CUSTOM_AUTH" in secret_env
        user = http.app.state.services.auth.authenticate(token)
        calls.append((project_id, asset_id, user.username, url))
        return {"pid": 123, "log": str(tmp_path / "launch.log"), "executable": "Slicer"}

    monkeypatch.setattr(ViewerManager, "launch", launch)
    job = client.post(f"/api/assets/{assets[0]['id']}/viewer?mode={mode}")
    result = client.wait(job["id"])
    assert result["pid"] == 123
    assert calls == [(setup["project_id"], assets[0]["id"], "owner", "http://localhost")]
    assert "token" not in json.dumps(result)


def test_viewer_receives_session_without_provider_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NV_INFERENCE_API_KEY", "provider-secret")
    monkeypatch.setenv("CUSTOM_AUTH", "custom-provider-secret")
    monkeypatch.setenv("MONAILABEL_TOKEN", "parent-session")
    monkeypatch.setenv("DISPLAY", ":42")
    launches = []

    def start(command, **kwargs):
        launches.append((command, kwargs["env"]))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("monailabel.viewers.manager.subprocess.Popen", start)
    result = ViewerManager(root=tmp_path).launch(
        Installation("slicer", "/opt/Slicer", "system", "existing"),
        "http://localhost:8000",
        "project",
        asset_id="sample",
        token="viewer-session",
        secret_env={"CUSTOM_AUTH"},
    )
    command, env = launches[0]
    assert command[0] == "/opt/Slicer"
    assert env["DISPLAY"] == ":42"
    assert not {"NV_INFERENCE_API_KEY", "CUSTOM_AUTH", "MONAILABEL_TOKEN"} & env.keys()
    config = Path(env["MONAILABEL_VIEWER_SESSION"])
    assert config.stat().st_mode & 0o777 == 0o600
    assert json.loads(config.read_text()) == {
        "url": "http://localhost:8000",
        "project_id": "project",
        "asset_id": "sample",
        "mode": "annotation",
        "shared_filesystem": False,
        "token": "viewer-session",
    }
    assert "viewer-session" not in json.dumps(result)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("flip_rows", [False, True])
@pytest.mark.parametrize("flip_columns", [False, True])
def test_plane_transform_is_lossless_for_asymmetric_arrays(transpose, flip_rows, flip_columns):
    from monailabel.core.geometry import orient_plane, restore_plane
    from monailabel.core.models import PlaneOrientation

    image = np.arange(30, dtype=np.float32).reshape((3, 5, 2))
    orientation = PlaneOrientation(
        transpose=transpose, flip_rows=flip_rows, flip_columns=flip_columns
    )
    displayed = orient_plane(image, orientation)
    assert displayed.shape == ((5, 3, 2) if transpose else (3, 5, 2))
    np.testing.assert_array_equal(restore_plane(displayed, orientation), image)


def test_slicer_radiological_orientation_and_in_plane_rotation():
    xy_to_ras = np.diag([-1.0, 1.0, 1.0, 1.0])
    xy_to_ras[2, 3] = 5
    scope = source_slice(np.eye(4), xy_to_ras, [12, 13, 14], [-160, 240])
    assert scope["orientation"] == {"transpose": True, "flip_rows": True, "flip_columns": True}
    rotated = np.eye(4)
    rotated[:2, :2] = [[0.707, -0.707], [0.707, 0.707]]
    with pytest.raises(ValueError, match="in-plane"):
        source_slice(np.eye(4), rotated, [12, 13, 14], [-160, 240])


def test_qupath_uses_its_own_installation_and_cache(tmp_path, monkeypatch):
    from monailabel.viewers.manager import QUPATH_LINUX

    content = archive("QuPath/bin/QuPath")
    spec = replace(QUPATH_LINUX, checksum=hashlib.sha512(content).hexdigest())
    monkeypatch.setattr("monailabel.viewers.provisioning.desktop.release", lambda name: spec)
    monkeypatch.setattr(DesktopProvisioner, "discover", lambda self, viewer: None)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **kw: httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
        ).stream("GET", "https://download.example/qupath"),
    )
    manager = ViewerManager(root=tmp_path)
    first = manager.ensure("qupath", progress=lambda _: None)
    again = manager.ensure("qupath", progress=lambda _: None)
    assert first.viewer == "qupath" and again.source == "cache"
    assert first.executable == again.executable
    assert Path(first.executable).name == "QuPath"


def test_ohif_static_routes_preserve_security_boundary(http, tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_OHIF_DIST", str(tmp_path / "ohif"))
    assert http.get("/ohif/monailabel").status_code == 503
    root = tmp_path / "ohif"
    root.mkdir()
    (root / "index.html").write_text("<html>OHIF fixture</html>")
    response = http.get("/ohif/monailabel")
    assert response.status_code == 200 and "OHIF fixture" in response.text
    assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert "'wasm-unsafe-eval'" in response.headers["Content-Security-Policy"]
    assert "'wasm-unsafe-eval'" not in http.get("/").headers["Content-Security-Policy"]
    assert http.get("/ohif/missing.js").status_code == 404
    http.cookies.clear()
    assert http.get("/api/dicomweb/studies").status_code == 401


def test_windows_portable_recipe_download_and_reuse(tmp_path, monkeypatch):
    import zipfile

    from monailabel.viewers.provisioning.catalog import release

    spec = release("qupath", "Windows", "AMD64")
    assert spec is not None and spec.archive == "zip"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(spec.executable, b"Windows executable fixture")
    content = stream.getvalue()
    spec = replace(spec, checksum=hashlib.sha256(content).hexdigest())
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr(DesktopProvisioner, "discover", lambda self, viewer: None)
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **kw: httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
        ).stream("GET", "https://download.example/qupath"),
    )
    provisioner = DesktopProvisioner(tmp_path)
    first = provisioner.ensure("qupath", spec=spec)
    second = provisioner.ensure("qupath", spec=spec)
    assert first.source == "download" and second.source == "cache"
    assert first.executable == second.executable


@pytest.mark.parametrize("name", ["../escape", r"..\escape", "C:/escape", "/escape", "safe:stream"])
def test_zip_rejects_cross_platform_unsafe_paths(tmp_path, name):
    import zipfile

    from monailabel.viewers.provisioning.desktop import extract

    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as bundle:
        bundle.writestr(name, b"unsafe")
    with pytest.raises(DomainError, match="unsafe path"):
        extract(archive_path, tmp_path / "out", "zip")


def test_prepared_ohif_distribution_does_not_require_a_build(tmp_path, monkeypatch):
    from monailabel.viewers.ohif import OhifManager

    dist = tmp_path / "prepared"
    dist.mkdir()
    (dist / "index.html").write_text("Prepared MONAI Label OHIF fixture")
    monkeypatch.setenv("MONAILABEL_OHIF_DIST", str(dist))
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert OhifManager(tmp_path / "empty-cache").ensure() == dist
    assert not (tmp_path / "empty-cache").exists()
