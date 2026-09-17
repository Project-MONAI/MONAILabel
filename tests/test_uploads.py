import gzip
import io

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.server.data import decode_image


@pytest.fixture
def upload_project(client):
    return client.post(
        "/api/projects",
        {
            "name": "Upload regression",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
            ],
        },
    )


def test_binary_upload_preserves_large_volume_and_retries_are_idempotent(http, upload_project):
    # Just beyond the former 16-million-voxel cap, with an asymmetric source affine.
    values = np.zeros((512, 512, 65), dtype=np.float32)
    values[12, 34, 56] = 71.5
    affine = np.diag([-0.75, -0.75, 3.0, 1.0])
    affine[:3, 3] = [123, 456, -50]
    volume = nib.Nifti1Image(values, affine)
    volume.header.set_xyzt_units("mm")
    source = volume.to_bytes()
    compressed = gzip.compress(source, compresslevel=1)
    path = f"/api/projects/{upload_project['id']}/assets/upload"
    params = {"name": "scan.nii.gz", "group_id": "patient-1", "split": "train"}
    response = http.post(path, params=params, content=compressed)
    assert response.status_code == 201, response.text
    asset = response.json()
    assert asset["spatial_shape"] == [512, 512, 65]
    np.testing.assert_allclose(asset["affine"], affine)
    assert http.get(f"/api/assets/{asset['id']}/image").content == source
    stored = http.app.state.services.artifacts.array(asset["image_key"])
    assert isinstance(stored, np.memmap) and not stored.flags.writeable
    np.testing.assert_array_equal(stored[..., 0], values)
    assert http.post(path, params=params, content=compressed).json()["id"] == asset["id"]
    assert len(http.get(f"/api/projects/{upload_project['id']}/assets").json()) == 1
    params["split"] = "validation"
    assert http.post(path, params=params, content=compressed).status_code == 409


def test_binary_upload_size_is_bounded_with_or_without_content_length(
    http, upload_project, monkeypatch
):
    monkeypatch.setattr("monailabel.server.uploads.MAX_FILE_BYTES", 32)
    path = f"/api/projects/{upload_project['id']}/assets/upload"
    params = {"name": "scan.nii", "group_id": "patient-1"}
    for content in (b"x" * 33, iter([b"x" * 16, b"x" * 17])):
        response = http.post(path, params=params, content=content)
        assert response.status_code == 413
    assert http.post(path, params=params, content=b"").status_code == 422
    assert http.get(f"/api/projects/{upload_project['id']}/assets").json() == []


def test_invalid_file_does_not_prevent_next_upload(http, upload_project):
    path = f"/api/projects/{upload_project['id']}/assets/upload"
    params = {"name": "scan.nii.gz", "group_id": "patient-1"}
    response = http.post(path, params=params, content=b"not an image")
    assert response.status_code == 422
    source = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.int16), np.eye(4)).to_bytes()
    response = http.post(path, params=params, content=gzip.compress(source))
    assert response.status_code == 201


def test_automatic_import_groups_keep_duplicate_images_in_one_split(http, upload_project):
    path = f"/api/projects/{upload_project['id']}/assets/upload"
    source = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.int16), np.eye(4)).to_bytes()
    first = http.post(path, params={"name": "scan.nii", "split": "train"}, content=source)
    assert first.status_code == 201
    renamed = http.post(path, params={"name": "renamed.nii", "split": "train"}, content=source)
    assert renamed.status_code == 201
    assert renamed.json()["group_id"] == first.json()["group_id"]
    conflict = http.post(
        path, params={"name": "renamed.nii", "split": "validation"}, content=source
    )
    assert conflict.status_code == 409
    other_source = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.int16), np.eye(4)).to_bytes()
    other = http.post(path, params={"name": "scan.nii", "split": "train"}, content=other_source)
    assert other.status_code == 201
    assert other.json()["group_id"] != first.json()["group_id"]
    named_source = nib.Nifti1Image(np.full((2, 3, 4), 2, dtype=np.int16), np.eye(4)).to_bytes()
    named = http.post(
        path, params={"name": "named.nii", "group_id": "patient-1"}, content=named_source
    )
    assert named.status_code == 201
    automatic = http.post(path, params={"name": "another-name.nii"}, content=named_source)
    assert automatic.status_code == 201
    assert automatic.json()["group_id"] == "patient-1"


def test_decompression_and_declared_shape_limits_are_checked_before_materializing(monkeypatch):
    monkeypatch.setattr("monailabel.server.data.MAX_NIFTI_BYTES", 128)
    with pytest.raises(DomainError, match="Decompressed"):
        decode_image("bomb.nii.gz", gzip.compress(b"x" * 129))
    header = nib.Nifti1Header()
    header.set_data_shape((512, 512, 257))
    header.set_data_dtype(np.float32)
    stream = io.BytesIO()
    header.write_to(stream)
    stream.write(b"\0" * 4)
    with pytest.raises(DomainError, match="exceeds.*voxels"):
        decode_image("oversized.nii", stream.getvalue())


def test_binary_upload_requires_manager(http, client, upload_project):
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{upload_project['id']}/members",
        {"user_id": user["id"], "roles": ["annotator"]},
    )
    http.post("/api/auth/login", json={"username": "annotator", "password": "test-password-1234"})
    response = http.post(
        f"/api/projects/{upload_project['id']}/assets/upload",
        params={"name": "scan.nii", "group_id": "patient-1"},
        content=b"image",
    )
    assert response.status_code == 403
