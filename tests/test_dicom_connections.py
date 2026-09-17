import io
import json
from email import policy
from email.parser import BytesParser

import httpx
import nibabel as nib
import numpy as np
import pydicom
import pytest

from monailabel.core.dicom import DicomFilters, DicomSeriesRef
from monailabel.core.errors import DomainError
from monailabel.core.models import DicomSeries
from monailabel.dicom.nifti_view import viewing_series
from monailabel.dicom.series import derived_ct_series
from monailabel.dicom.web import DicomwebClient, query


def archive():
    contents = derived_ct_series(
        np.arange(60, dtype=np.float32).reshape(5, 4, 3), np.eye(4), "Archive fixture"
    )
    first = pydicom.dcmread(io.BytesIO(contents[0]))
    metadata = first.to_json_dict()
    metadata.pop("7FE00010")
    metadata["00201209"] = {"vr": "IS", "Value": [3]}
    ref = DicomSeriesRef(
        study_uid=str(first.StudyInstanceUID), series_uid=str(first.SeriesInstanceUID)
    )
    return contents, metadata, ref


@pytest.fixture
def remote(monkeypatch):
    contents, metadata, ref = archive()
    calls = []

    def get(self, path, *, params=None, **kwargs):
        calls.append((path, params, self.headers))
        if path == "/series":
            rows = [metadata] if int((params or {}).get("offset", 0)) == 0 else []
            return json.dumps(rows).encode(), "application/dicom+json"
        if path.endswith("/instances"):
            rows = [
                {
                    "00080018": {
                        "vr": "UI",
                        "Value": [str(pydicom.dcmread(io.BytesIO(c)).SOPInstanceUID)],
                    }
                }
                for c in contents
            ]
            return json.dumps(
                rows if int(params.get("offset", 0)) == 0 else []
            ).encode(), "application/dicom+json"
        uid = path.rsplit("/", 1)[-1]
        return next(
            c for c in contents if str(pydicom.dcmread(io.BytesIO(c)).SOPInstanceUID) == uid
        ), "application/dicom"

    monkeypatch.setattr(DicomwebClient, "get", get)
    return contents, ref, calls


def test_filter_mapping_and_invalid_dates():
    filters = DicomFilters(
        modality="MR",
        date_from="2026-01-02",
        date_to="2026-03-04",
        patient_id="P01",
        patient_name="Smith",
        study_description="Abdomen",
        series_description="T2",
        accession_number="A1",
    )
    params = query(filters)
    assert params["Modality"] == "MR" and params["StudyDate"] == "20260102-20260304"
    assert params["PatientID"] == "P01" and params["PatientName"] == "*Smith*"
    assert params["StudyDescription"] == "*Abdomen*" and params["SeriesDescription"] == "*T2*"
    assert params["AccessionNumber"] == "A1"
    with pytest.raises(ValueError):
        DicomFilters(date_from="2026-03-04", date_to="2026-01-02")
    with pytest.raises(ValueError):
        DicomSeriesRef(study_uid="..", series_uid="1.2")


def test_connect_search_import_and_local_serving_without_origin(client, http, remote, monkeypatch):
    contents, ref, calls = remote
    project = client.post("/api/projects", {"name": "DICOM source"})
    prefix = f"/api/projects/{project['id']}/dicom-connections"
    connection = client.post(
        prefix,
        {
            "name": "Test archive",
            "url": "https://archive.example/dicom-web",
            "authentication": "basic",
            "username": "user",
            "password": "private-password",
        },
    )
    assert "private-password" not in json.dumps(connection)
    assert calls[-1][2]["Authorization"].startswith("Basic ")
    path = prefix + "/" + connection["id"]
    result = client.post(path + "/search", {"modality": "CT"})
    assert len(result["items"]) == 1 and not result["truncated"]
    imported = client.wait(client.post(path + "/imports", {"series": [ref.model_dump()]})["id"])
    assert len(imported["asset_ids"]) == 1 and not imported["failed"]
    asset_id = imported["asset_ids"][0]
    asset = client.get("/api/assets/" + asset_id)
    assert asset["spatial_shape"] == [5, 4, 3]
    assert client.post(path + "/search", {})["items"] == []
    assert (
        client.post(path + "/search", {"import_status": "imported"})["items"][0][
            "imported_asset_id"
        ]
        == asset_id
    )
    retry = client.wait(client.post(path + "/imports", {"series": [ref.model_dump()]})["id"])
    assert retry["skipped_asset_ids"] == [asset_id] and retry["asset_ids"] == []

    def unavailable(*args, **kwargs):
        raise AssertionError("Viewing must not contact the origin")

    monkeypatch.setattr(DicomwebClient, "get", unavailable)
    monkeypatch.setattr(http.app.state.services.dicom, "request", unavailable)
    rows = client.get("/api/dicomweb/studies")
    assert rows[0]["0020000D"]["Value"] == [ref.study_uid]
    base = f"/api/dicomweb/studies/{ref.study_uid}/series/{ref.series_uid}"
    metadata = client.get(base + "/metadata")
    assert len(metadata) == 3
    uid = str(pydicom.dcmread(io.BytesIO(contents[0])).SOPInstanceUID)
    response = http.get(base + f"/instances/{uid}/frames/1")
    assert response.status_code == 200
    mime = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {response.headers['content-type']}\r\n\r\n".encode() + response.content
    )
    frame = list(mime.iter_parts())[0].get_payload(decode=True)
    np.testing.assert_array_equal(
        np.frombuffer(frame, dtype="<i2").reshape(4, 5),
        pydicom.dcmread(io.BytesIO(contents[0])).pixel_array,
    )
    assert http.get(base + "/instances/9.9.9/frames/1").status_code == 403
    assert (
        http.get(f"/api/dicomweb/studies/{ref.study_uid}/series/9.9.9/metadata").status_code == 403
    )
    user = client.post("/api/auth/users", {"username": "outsider", "password": "outsider-password"})
    assert user["id"]
    http.post("/api/auth/login", json={"username": "outsider", "password": "outsider-password"})
    assert http.get(base + "/metadata").status_code == 403
    assert http.post(path + "/search", json={}).status_code == 403


def test_connection_permissions_and_cross_project_ids(client, http, remote):
    first = client.post("/api/projects", {"name": "First"})
    other = client.post("/api/projects", {"name": "Other"})
    connection = client.post(
        f"/api/projects/{first['id']}/dicom-connections",
        {"name": "Archive", "url": "https://archive.example/dicom-web"},
    )
    assert (
        http.post(
            f"/api/projects/{other['id']}/dicom-connections/{connection['id']}/search", json={}
        ).status_code
        == 403
    )
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "annotator-password"}
    )
    client.request(
        "PUT",
        f"/api/projects/{first['id']}/members",
        {"user_id": user["id"], "roles": ["annotator"]},
    )
    http.post("/api/auth/login", json={"username": "annotator", "password": "annotator-password"})
    assert (
        http.post(
            f"/api/projects/{first['id']}/dicom-connections",
            json={"name": "Archive", "url": "https://archive.example"},
        ).status_code
        == 403
    )


def test_dicomweb_handles_multipart_auth_errors_and_size_bounds():
    contents, _, ref = archive()
    uid = str(pydicom.dcmread(io.BytesIO(contents[0])).SOPInstanceUID)

    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path.endswith("/instances"):
            return httpx.Response(
                200,
                json=[]
                if request.url.params.get("offset") != "0"
                else [{"00080018": {"Value": [uid]}}],
            )
        body = (
            b"--test\r\nContent-Type: application/dicom\r\n\r\n" + contents[0] + b"\r\n--test--\r\n"
        )
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": 'multipart/related; type="application/dicom"; boundary=test'},
        )

    source = DicomwebClient(
        "https://source.example",
        {"Authorization": "Bearer test-token"},
        transport=httpx.MockTransport(respond),
    )
    assert source.instances(ref, lambda part: None) == [contents[0]]
    with pytest.raises(DomainError, match="size limit"):
        source.get("/file", limit=3)
    source = DicomwebClient(
        "https://source.example", transport=httpx.MockTransport(lambda request: httpx.Response(401))
    )
    with pytest.raises(DomainError, match="credentials"):
        source.check()


@pytest.mark.parametrize("floating,reverse", [(False, False), (True, False), (True, True)])
def test_nifti_viewing_copy_preserves_geometry_and_intensity_with_bounded_rounding(
    floating, reverse
):
    image = np.arange(60, dtype=np.float32).reshape(5, 4, 3) - 20
    if floating:
        image = image / 7.1
    affine = np.diag([-0.8, -1.2, -3 if reverse else 3, 1])
    affine[:3, 3] = [125, -90, 35]
    contents = viewing_series(image, affine, "NIfTI case", "case-id", lambda part: None)
    for index, content in enumerate(contents):
        ds = pydicom.dcmread(io.BytesIO(content))
        assert ds.Modality == "OT" and str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.7"
        restored = ds.pixel_array.T * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        np.testing.assert_allclose(
            restored, image[:, :, index], atol=float(ds.RescaleSlope) / 2 + 1e-6
        )
        lps = np.diag([-1, -1, 1, 1]) @ affine
        np.testing.assert_allclose(ds.ImagePositionPatient, (lps @ [0, 0, index, 1])[:3])
        np.testing.assert_allclose(ds.ImageOrientationPatient, [1, 0, 0, 0, 1, 0])


def test_nifti_ohif_cache_keeps_original_asset_and_reuses_dicom(client, http, monkeypatch):
    monkeypatch.setattr("monailabel.viewers.ohif.OhifManager.ensure", lambda *args: None)
    project = client.post("/api/projects", {"name": "NIfTI in OHIF"})
    source = nib.Nifti1Image(
        np.arange(60, dtype=np.float32).reshape(5, 4, 3) / 3.7, np.eye(4)
    ).to_bytes()
    response = http.post(
        f"/api/projects/{project['id']}/assets/upload", params={"name": "case.nii"}, content=source
    )
    assert response.status_code == 201
    asset = response.json()
    first = client.wait(client.post(f"/api/assets/{asset['id']}/viewer?name=ohif", {})["id"])
    record = client.get(f"/api/assets/{asset['id']}/dicom")
    assert record["derived_from_nifti"]
    again = client.wait(client.post(f"/api/assets/{asset['id']}/viewer?name=ohif", {})["id"])
    assert first["url"] == again["url"]
    assert len(http.app.state.services.store.list(DicomSeries, project["id"])) == 1
    assert client.get(f"/api/assets/{asset['id']}") == asset
    assert http.get(f"/api/assets/{asset['id']}/image").content == source


def test_ohif_study_browser_is_scoped_to_the_selected_asset(client, http, monkeypatch):
    monkeypatch.setattr("monailabel.viewers.ohif.OhifManager.ensure", lambda *args: None)
    source = nib.Nifti1Image(np.zeros((5, 4, 3), dtype=np.float32), np.eye(4)).to_bytes()
    assets, series = [], []
    # The same source may legitimately exist in separate annotation projects.
    for name in ("Annotation", "Comparison"):
        project = client.post("/api/projects", {"name": name})
        asset = http.post(
            f"/api/projects/{project['id']}/assets/upload",
            params={"name": "case.nii"},
            content=source,
        ).json()
        client.wait(client.post(f"/api/assets/{asset['id']}/viewer?name=ohif", {})["id"])
        assets.append(asset)
        series.append(client.get(f"/api/assets/{asset['id']}/dicom"))
    assert len(client.get("/api/dicomweb/studies")) == 2
    for asset, own, other in [(assets[0], series[0], series[1]), (assets[1], series[1], series[0])]:
        base = f"/api/assets/{asset['id']}/dicomweb"
        rows = client.get(base + "/studies")
        assert [r["0020000D"]["Value"][0] for r in rows] == [own["study_uid"]]
        assert len(client.get(base + f"/studies/{own['study_uid']}/metadata")) == 3
        assert http.get(base + f"/studies/{other['study_uid']}/metadata").status_code == 403
        assert client.get(base + "/studies?StudyInstanceUID=" + other["study_uid"]) == []
    client.post("/api/auth/users", {"username": "outsider", "password": "outsider-password"})
    client.post("/api/auth/login", {"username": "outsider", "password": "outsider-password"})
    assert http.get(base + "/studies").status_code == 403
    client.post("/api/auth/logout")
    assert http.get(base + "/studies").status_code == 401


def test_bounded_search_and_nonadvancing_pages(monkeypatch):
    _, row, _ = archive()
    client = DicomwebClient("https://archive.example")

    def pages(path, params):
        start = int(params["offset"])
        return [
            row | {"0020000E": {"vr": "UI", "Value": [f"1.2.{index}"]}}
            for index in range(start, start + 100)
        ]

    monkeypatch.setattr(client, "rows", pages)
    records, truncated = client.search(DicomFilters())
    assert len(records) == 1000 and truncated
    monkeypatch.setattr(client, "rows", lambda *a: [row])
    with pytest.raises(DomainError, match="not advancing"):
        client.search(DicomFilters())


def test_batch_partial_failure_keeps_completed_series(client, http, remote, monkeypatch):
    from monailabel.core.models import Asset

    _, ref, _ = remote
    project = client.post("/api/projects", {"name": "Partial import"})
    prefix = f"/api/projects/{project['id']}/dicom-connections"
    connection = client.post(prefix, {"name": "Archive", "url": "https://archive.example"})
    original = DicomwebClient.instances

    def instances(self, selected, progress):
        if selected.series_uid == "9.9":
            raise DomainError("Archive instance unavailable")
        return original(self, selected, progress)

    monkeypatch.setattr(DicomwebClient, "instances", instances)
    result = client.wait(
        client.post(
            prefix + "/" + connection["id"] + "/imports",
            {
                "series": [ref.model_dump(), {"study_uid": "9.8", "series_uid": "9.9"}],
            },
        )["id"]
    )
    assert len(result["asset_ids"]) == 1 and len(result["failed"]) == 1
    assert len(http.app.state.services.store.list(Asset, project["id"])) == 1


@pytest.mark.parametrize("split", ["pool", "train"])
def test_dicom_duplicate_image_respects_existing_group_and_learning_split(
    client, http, remote, split
):
    from monailabel.dicom.series import read_series

    contents, ref, _ = remote
    project = client.post("/api/projects", {"name": "Same source in two formats"})
    prefix = f"/api/projects/{project['id']}"
    volume = read_series(contents)
    source = nib.Nifti1Image(volume.image, volume.affine).to_bytes()
    response = http.post(
        prefix + "/assets/upload",
        params={"name": "same.nii", "group_id": "known-patient", "split": split},
        content=source,
    )
    assert response.status_code == 201
    connection = client.post(
        prefix + "/dicom-connections", {"name": "Archive", "url": "https://archive.example"}
    )
    result = client.wait(
        client.post(
            prefix + "/dicom-connections/" + connection["id"] + "/imports",
            {"series": [ref.model_dump()]},
        )["id"]
    )
    if split == "train":
        assert not result["asset_ids"] and "learning split" in result["failed"][0]["error"]
    else:
        imported = client.get("/api/assets/" + result["asset_ids"][0])
        assert imported["group_id"] == "known-patient"
