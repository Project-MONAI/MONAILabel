"""Shared-file loading preserves image geometry, immutable sources and review freshness."""

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from monailabel.viewers.resources.volume_files import local_nifti


@pytest.mark.parametrize("compressed", [False, True])
def test_local_volume_preserves_source_and_reuses_private_copy(tmp_path, compressed):
    values = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    affine = np.diag([-0.7, 0.8, 3.0, 1.0])
    affine[:3, 3] = [27, 41, -55]
    data = nib.Nifti1Image(values, affine).to_bytes()
    original = tmp_path / "workspace.bin"
    original.write_bytes(gzip.compress(data) if compressed else data)
    cache = tmp_path / "viewer"
    cache.mkdir()
    path = local_nifti(str(original), cache)
    assert path is not None
    assert path.name.endswith(".nii.gz" if compressed else ".nii")
    volume = nib.load(path)
    np.testing.assert_array_equal(volume.get_fdata(), values)
    np.testing.assert_allclose(volume.affine, affine)
    modified = path.stat().st_mtime_ns
    assert local_nifti(str(original), cache) == path
    assert path.stat().st_mtime_ns == modified
    assert not path.samefile(original)
    source_bytes = original.read_bytes()
    path.write_bytes(b"Native viewer save")
    assert original.read_bytes() == source_bytes


def test_unavailable_local_source_or_failed_copy_uses_download(tmp_path, monkeypatch):
    assert local_nifti(None, tmp_path) is None
    assert local_nifti(str(tmp_path / "missing"), tmp_path) is None
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")

    def interrupted(source, target):
        Path(target).write_bytes(b"partial copy")
        raise OSError("Filesystem unavailable")

    monkeypatch.setattr("monailabel.viewers.resources.volume_files.shutil.copyfile", interrupted)
    assert local_nifti(str(source), tmp_path) is None
    assert list(tmp_path.iterdir()) == [source]


def test_shared_files_resolve_latest_mask_and_authorize_each_open(client, http, seeded):
    _, assets = seeded
    asset = assets[0]
    path = f"/api/assets/{asset['id']}/viewer-files"
    files = client.get(path)
    assert files["asset"] == asset and files["mask_path"] is None
    assert Path(files["image_path"]).is_absolute()
    assert (
        Path(files["image_path"]).read_bytes()
        == http.get(f"/api/assets/{asset['id']}/image").content
    )
    mask = np.asarray(client.get(f"/api/assets/{asset['id']}/fixture")["mask"], dtype=np.uint8)
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "covered_labels": [0, 1, 2], "mask": mask.tolist()},
    )
    submitted = client.get(path)
    assert submitted["asset"]["annotation_id"] == annotation["id"]
    original_mask = np.load(submitted["mask_path"], mmap_mode="r", allow_pickle=False)
    assert not original_mask.flags.writeable
    np.testing.assert_array_equal(original_mask, mask)
    mask[0, 0, 0] = 1
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 1, "covered_labels": [0, 1, 2], "mask": mask.tolist()},
    )
    updated = client.get(path)
    assert updated["asset"]["revision"] == 2
    assert updated["image_path"] == submitted["image_path"]
    assert updated["mask_path"] != submitted["mask_path"]
    np.testing.assert_array_equal(np.load(updated["mask_path"], allow_pickle=False), mask)
    # Reuse the running test app; do not enter a second application lifespan.
    remote = TestClient(http.app, client=("192.0.2.1", 1234), cookies=http.cookies)
    try:
        assert remote.get(path).status_code == 403
    finally:
        remote.close()
    client.post("/api/auth/users", {"username": "outsider", "password": "outsider-password"})
    client.post("/api/auth/login", {"username": "outsider", "password": "outsider-password"})
    assert http.get(path).status_code == 403
    client.post("/api/auth/logout")
    assert http.get(path).status_code == 401
