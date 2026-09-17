import numpy as np

from monailabel.server.data import nifti_bytes


def test_mask_import_geometry_and_review_boundary(client, http, seeded):
    setup, assets = seeded
    asset = assets[0]
    mask = np.asarray(client.get(f"/api/assets/{asset['id']}/fixture")["mask"], dtype=np.uint8)
    path = f"/api/assets/{asset['id']}/mask-import"
    wrong = np.asarray(asset["affine"])
    wrong[0, 3] += 5
    response = http.post(
        path,
        params={"name": "mask.nii", "base_revision": 0},
        content=nifti_bytes(mask, wrong.tolist()),
    )
    assert response.status_code == 422
    assert "affine" in response.text
    response = http.post(
        path,
        params={"name": "mask.nii", "base_revision": 0},
        content=nifti_bytes(mask, asset["affine"]),
    )
    assert response.status_code == 201
    assert response.json()["covered_labels"] == [0, 1, 2]
    assert http.post(f"/api/projects/{setup['project_id']}/snapshots").status_code == 422


def test_source_header_and_integer_export_preserve_spatial_metadata(client, http):
    import base64
    import gzip

    import nibabel as nib

    project = client.post(
        "/api/projects",
        {
            "name": "Geometry regression",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Target", "color": "#ff0000"},
            ],
        },
    )
    affine = np.array([[0, -1.5, 0, 30], [2, 0, 0, -40], [0, 0, 3, 10], [0, 0, 0, 1.0]])
    source = nib.Nifti1Image(np.arange(24, dtype=np.int16).reshape((2, 3, 4)), affine)
    source.set_qform(affine, code=1)
    source.set_sform(affine, code=4)
    source.header.set_xyzt_units("mm", "sec")
    content = source.to_bytes()
    asset = client.post(
        f"/api/projects/{project['id']}/assets",
        {
            "name": "source.nii.gz",
            "group_id": "patient-1",
            "split": "train",
            "image_base64": base64.b64encode(gzip.compress(content)).decode(),
        },
    )
    assert http.get(f"/api/assets/{asset['id']}/image").content == content
    mask = (np.arange(24).reshape((2, 3, 4)) > 10).astype(np.uint8)
    client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "covered_labels": [0, 1],
            "mask": mask.tolist(),
        },
    )
    exported = nib.Nifti1Image.from_bytes(
        http.get(f"/api/assets/{asset['id']}/segmentation.nii").content
    )
    assert exported.header.get_xyzt_units() == ("mm", "sec")
    assert int(exported.header["qform_code"]) == 1
    assert int(exported.header["sform_code"]) == 4
    assert exported.get_data_dtype() == np.uint8
    np.testing.assert_allclose(exported.affine, affine)
    np.testing.assert_array_equal(np.asarray(exported.dataobj), mask)
