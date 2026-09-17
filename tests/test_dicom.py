import io

import httpx
import numpy as np
import pydicom
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, DicomSeries
from monailabel.dicom.series import derived_ct_series, read_series
from monailabel.server.dicom import Dicom


def sample_series():
    image = np.arange(7 * 5 * 4, dtype=np.float32).reshape(7, 5, 4) - 100
    affine = np.diag([-0.8, -1.2, 3, 1])
    affine[:3, 3] = [-120, 70, -20]
    return image, affine, derived_ct_series(image, affine, "Geometry fixture")


def modified(content, **attributes):
    ds = pydicom.dcmread(io.BytesIO(content))
    for key, value in attributes.items():
        setattr(ds, key, value)
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


@pytest.mark.parametrize("reverse", [False, True])
def test_dicom_preserves_voxels_and_patient_coordinates(reverse):
    image, affine, contents = sample_series()
    if reverse:
        affine[:3, 2] *= -1
        contents = derived_ct_series(image, affine, "Reversed source K")
    volume = read_series(contents[::-1])
    expected = image[:, :, ::-1] if reverse else image
    np.testing.assert_array_equal(volume.image, expected)
    for i, j, k in [(0, 0, 0), (6, 4, 3), (2, 3, 1)]:
        source_k = image.shape[2] - k - 1 if reverse else k
        np.testing.assert_allclose(volume.affine @ [i, j, k, 1], affine @ [i, j, source_k, 1])
    assert volume.instance_uids == [
        str(pydicom.dcmread(io.BytesIO(c)).SOPInstanceUID)
        for c in (contents[::-1] if reverse else contents)
    ]


@pytest.mark.parametrize("change", ["mixed", "irregular", "tilted", "duplicate", "multiframe"])
def test_dicom_rejects_ambiguous_source_geometry(change):
    _, _, contents = sample_series()
    if change == "mixed":
        contents[0] = modified(contents[0], SeriesInstanceUID=pydicom.uid.generate_uid())
    elif change == "irregular":
        contents[-1] = modified(contents[-1], ImagePositionPatient=[120, -70, -9])
    elif change == "tilted":
        contents[-1] = modified(contents[-1], ImagePositionPatient=[121, -70, -11])
    elif change == "duplicate":
        contents.append(contents[0])
    else:
        contents[0] = modified(contents[0], NumberOfFrames=2)
    with pytest.raises(DomainError):
        read_series(contents)


def test_dicom_applies_rescale_and_rejects_fractional_test_hu():
    image, affine, contents = sample_series()
    volume = read_series([modified(c, RescaleSlope=2, RescaleIntercept=-1024) for c in contents])
    np.testing.assert_array_equal(volume.image, image * 2 - 1024)
    with pytest.raises(DomainError, match="integer HU"):
        derived_ct_series(image + 0.25, affine, "No rounding")


def mock_orthanc(monkeypatch, contents):
    def request(self, method, path, **kwargs):
        if path == "/tools/find":
            return httpx.Response(200, json=["series"])
        if path == "/series/series/instances":
            return httpx.Response(200, json=[{"ID": str(i)} for i in range(len(contents))])
        if path.startswith("/instances/"):
            return httpx.Response(200, content=contents[int(path.split("/")[2])])
        raise AssertionError(path)

    monkeypatch.setattr(Dicom, "request", request)


def test_import_is_atomic_and_retains_dicom_source(client, http, seeded, monkeypatch):
    setup, _ = seeded
    image, affine, contents = sample_series()
    mock_orthanc(monkeypatch, contents)
    uid = str(pydicom.dcmread(io.BytesIO(contents[0])).SeriesInstanceUID)
    path = f"/api/projects/{setup['project_id']}/dicom-series"
    result = client.wait(client.post(path, {"series_uid": uid})["id"])
    asset = client.get("/api/assets/" + result["asset_id"])
    source = client.get("/api/assets/" + asset["id"] + "/dicom")
    assert asset["spatial_shape"] == list(image.shape) and asset["split"] == "pool"
    np.testing.assert_allclose(asset["affine"], affine)
    assert len(source["source_keys"]) == 4
    assert client.wait(client.post(path, {"series_uid": uid})["id"])["asset_id"] == asset["id"]
    count = len(http.app.state.services.store.list(Asset))
    # A malformed new series never publishes either an asset or a source record.
    new_uid = pydicom.uid.generate_uid()
    bad = [modified(c, SeriesInstanceUID=new_uid) for c in contents]
    bad[0] = modified(bad[0], NumberOfFrames=2)
    mock_orthanc(monkeypatch, bad)
    with pytest.raises(RuntimeError):
        client.wait(client.post(path, {"series_uid": new_uid})["id"])
    assert len(http.app.state.services.store.list(Asset)) == count
    assert len(client.get(path)) == 1


def test_dicomweb_requires_project_access_and_filters_studies(http, seeded, monkeypatch):
    setup, assets = seeded
    service = http.app.state.services
    source = DicomSeries(
        project_id=setup["project_id"],
        asset_id=assets[0]["id"],
        study_uid="1.2.3",
        series_uid="1.2.4",
        frame_of_reference_uid="1.2.5",
        instance_uids=["1.2.6"],
        source_keys=[service.artifacts.put(sample_series()[2][0])],
        orthanc_series_id="series",
    )
    with service.store.transaction() as session:
        session.insert(source)
    calls = []

    def request(self, method, path, **kwargs):
        calls.append(path)
        return httpx.Response(
            200, json=[{"0020000D": {"Value": [uid]}} for uid in ["1.2.3", "9.9.9"]]
        )

    monkeypatch.setattr(Dicom, "request", request)
    r = http.get("/api/dicomweb/studies")
    assert r.status_code == 200 and len(r.json()) == 1
    assert http.get("/api/dicomweb/studies/9.9.9/metadata").status_code == 403
    assert http.get("/api/dicomweb/tools/find").status_code == 404
    assert len(calls) == 0
    http.cookies.clear()
    assert http.get("/api/dicomweb/studies").status_code == 401


def test_direct_import_groups_same_patient_across_studies(client, http, seeded, monkeypatch):
    from monailabel.core.models import Split
    from monailabel.server.data import decode_image

    setup, _ = seeded
    image, affine, contents = sample_series()
    mock_orthanc(monkeypatch, contents)
    uid = str(pydicom.dcmread(io.BytesIO(contents[0])).SeriesInstanceUID)
    path = f"/api/projects/{setup['project_id']}/dicom-series"
    result = client.wait(client.post(path, {"series_uid": uid})["id"])
    services = http.app.state.services
    first = services.store.get(Asset, result["asset_id"])
    assert first.group_id.startswith("dicom-patient:")
    # Import entry points must agree on the decoded-image identity, including array order.
    decoded, _ = decode_image("source.nii", services.artifacts.read(first.source_key))
    assert services.artifacts.put_array(decoded) == first.image_key
    services.datasets.assign_split(first.id, Split.TRAIN)
    # A different study from the same patient must not become a new independent pool case.
    second = derived_ct_series(image + 1, affine, "Geometry fixture")
    mock_orthanc(monkeypatch, second)
    uid = str(pydicom.dcmread(io.BytesIO(second[0])).SeriesInstanceUID)
    with pytest.raises(RuntimeError, match="patient already belongs"):
        client.wait(client.post(path, {"series_uid": uid})["id"])
    assert len(client.get(path)) == 1
