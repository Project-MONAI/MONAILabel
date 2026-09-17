import gzip

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.models import RegionProposal
from monailabel.core.ports import Prediction
from monailabel.providers.remote import RemoteSegmenter


@pytest.mark.parametrize("spelling", ["bounding", "bonding"])
def test_box_prompt_uses_free_target_without_changing_segmentation_protocol(
    client, http, monkeypatch, spelling
):
    project = client.post(
        "/api/projects",
        {
            "name": "Spleen",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
            ],
        },
    )
    image = np.zeros((4, 5, 3), dtype=np.float32)
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "test.nii.gz", "group_id": "case"},
        content=gzip.compress(nib.Nifti1Image(image, np.eye(4)).to_bytes()),
    ).json()
    model = client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "GPT Astra",
            "provider": "openai-chat-polygons",
            "label_ids": [0, 1],
            "config": {"url": "https://example.test/chat/completions", "model": "astra"},
        },
    )

    def predict(self, image, labels, prompt, record):
        assert labels[1].name == "liver"
        assert image.shape == (5, 4, 1)
        mask = np.zeros((5, 4), dtype=np.uint8)
        mask[1:4, 1:3] = 1
        return Prediction(mask)

    monkeypatch.setattr(RemoteSegmenter, "predict", predict)
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": f"draw a {spelling} box for liver in the current slice "
            "using GPT Astra model",
            "context": {
                "asset_id": asset["id"],
                "slice": {
                    "axis": 2,
                    "index": 1,
                    "window": [-1, 1],
                    "orientation": {"transpose": True},
                },
            },
        },
    )
    result = client.wait(reply["job_id"])
    region = client.get(f"/api/regions/{result['region_id']}")
    assert region["bounds"] == [[1, 1, 1], [2, 3, 1]]
    assert region["target"] == "liver"
    assert region["model_id"] == model["id"]
    assert client.get(f"/api/projects/{project['id']}") == project
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
    assert len(http.app.state.services.store.list(RegionProposal, project["id"])) == 1


@pytest.mark.parametrize(
    ("message", "target", "all_matches", "current_slice"),
    [
        ("remove the bounding box for liver", "liver", False, False),
        ("Please delete the bounding box for Liver in current slice.", "Liver", False, True),
        ("clear all bounding boxes for spleen", "spleen", True, False),
        ("remove all the bounding boxes for left kidney on this slice", "left kidney", True, True),
    ],
)
def test_remove_box_prompt_returns_local_action_without_inference(
    client, http, seeded, message, target, all_matches, current_slice
):
    demo, assets = seeded
    pid = demo["project_id"]
    before = client.get(f"/api/projects/{pid}/jobs")
    reply = client.post(
        f"/api/projects/{pid}/assistant",
        {
            "message": message,
            "context": {
                "asset_id": assets[0]["id"],
                "viewer_actions": ["remove_regions"],
                "slice": {"axis": 2, "index": 0},
            },
        },
    )
    assert reply["job_id"] is None
    assert reply["data"] == {
        "client_action": "remove_regions",
        "project_id": pid,
        "asset_id": assets[0]["id"],
        "target": target,
        "all_matches": all_matches,
        "slice": {
            "axis": 2,
            "index": 0,
            "window": None,
            "orientation": {"transpose": False, "flip_rows": False, "flip_columns": False},
        }
        if current_slice
        else None,
    }
    assert client.get(f"/api/projects/{pid}/jobs") == before


def test_removal_requires_viewer_capability_and_checks_project_and_roles(client, http, seeded):
    demo, assets = seeded
    pid = demo["project_id"]
    body = {
        "message": "remove the bounding box for liver",
        "context": {"asset_id": assets[0]["id"]},
    }
    reply = client.post(f"/api/projects/{pid}/assistant", body)
    assert "updated Slicer" in reply["message"]
    assert not reply["data"]
    other = client.post("/api/demo")
    body["context"]["viewer_actions"] = ["remove_regions"]
    assert http.post(f"/api/projects/{other['project_id']}/assistant", json=body).status_code == 422
    reviewer = client.post(
        "/api/auth/users", {"username": "box-reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT", f"/api/projects/{pid}/members", {"user_id": reviewer["id"], "roles": ["reviewer"]}
    )
    http.post(
        "/api/auth/login", json={"username": "box-reviewer", "password": "test-password-1234"}
    )
    assert http.post(f"/api/projects/{pid}/assistant", json=body).status_code == 403


@pytest.fixture
def roi_case(client, http):
    project = client.post(
        "/api/projects",
        {
            "name": "ROI range",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
            ],
        },
    )
    image = np.zeros((9, 10, 90), dtype=np.float32)
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "roi.nii.gz", "group_id": "case"},
        content=gzip.compress(nib.Nifti1Image(image, np.eye(4)).to_bytes()),
    ).json()
    model = client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "Selected vision",
            "provider": "openai-chat-polygons",
            "label_ids": [0, 1],
            "config": {"url": "https://example.test/chat/completions", "model": "test"},
        },
    )
    return project, asset, model


@pytest.mark.parametrize("axis,first,last", [(2, 70, 80), (0, 2, 4), (1, 2, 4)])
def test_roi_combines_every_requested_slice_in_source_geometry(
    client, http, monkeypatch, roi_case, axis, first, last
):
    from monailabel.core.geometry import orient_plane
    from monailabel.core.models import PlaneOrientation

    project, asset, model = roi_case
    calls = []
    orientation = PlaneOrientation(transpose=True, flip_rows=True, flip_columns=True)

    def predict(self, image, labels, prompt, record):
        index = first - 1 + len(calls)
        calls.append(index)
        assert record.id == model["id"]
        assert labels[1].name == "spleen"
        assert f"source slice {index + 1}" in prompt
        shape = [size for a, size in enumerate(asset["spatial_shape"]) if a != axis]
        mask = np.zeros(shape, dtype=np.uint8)
        # Two endpoints have distinct extents; empty middle slices must still be queried.
        if index == first - 1:
            mask[1:3, 2:4] = 1
        if index == last - 1:
            mask[3:6, 4:7] = 1
        return Prediction(orient_plane(mask, orientation))

    monkeypatch.setattr(RemoteSegmenter, "predict", predict)
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": f"add roi for spleen between slice {first} to {last}",
            "context": {
                "asset_id": asset["id"],
                "model_id": model["id"],
                "viewer_actions": ["roi"],
                "slice": {
                    "axis": axis,
                    "index": 0,
                    "window": [-1, 1],
                    "orientation": orientation.model_dump(),
                },
            },
        },
    )
    assert "numbered from 1, inclusive" in reply["message"]
    result = client.wait(reply["job_id"])
    region = client.get(f"/api/regions/{result['region_id']}")
    low, high = [1, 2], [5, 6]
    low.insert(axis, first - 1)
    high.insert(axis, last - 1)
    assert region["bounds"] == [low, high]
    assert region["end_index"] == last - 1
    assert region["detected_slices"] == 2
    assert calls == list(range(first - 1, last))
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
    assert client.get(f"/api/projects/{project['id']}") == project


@pytest.mark.parametrize("first,last", [(0, 80), (80, 70), (70, 91)])
def test_roi_invalid_range_creates_no_job(client, http, roi_case, first, last):
    project, asset, model = roi_case
    response = http.post(
        f"/api/projects/{project['id']}/assistant",
        json={
            "message": f"add roi for spleen between slice {first} to {last}",
            "context": {
                "asset_id": asset["id"],
                "model_id": model["id"],
                "viewer_actions": ["roi"],
                "slice": {"axis": 2, "index": 0, "window": [-1, 1]},
            },
        },
    )
    assert response.status_code == 422
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.mark.parametrize("outcome", ["empty", "failed", "cancelled"])
def test_roi_empty_failure_and_cancel_never_publish_partial_bounds(
    client, http, monkeypatch, roi_case, outcome
):
    import time

    from monailabel.core.errors import DomainError
    from monailabel.core.models import Job

    project, asset, model = roi_case
    service = http.app.state.services
    calls = []

    def predict(self, image, labels, prompt, record):
        calls.append(prompt)
        if len(calls) == 2 and outcome == "failed":
            raise DomainError("Test inference failed")
        if len(calls) == 2 and outcome == "cancelled":
            service.jobs.cancel(service.store.list(Job, project["id"])[0].id)
        return Prediction(np.full(image.shape[:2], 0 if outcome == "empty" else 1, dtype=np.uint8))

    monkeypatch.setattr(RemoteSegmenter, "predict", predict)
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "add roi for spleen between slice 70 to 80 using Selected vision model",
            "context": {
                "asset_id": asset["id"],
                "viewer_actions": ["roi"],
                "slice": {"axis": 2, "index": 0, "window": [-1, 1]},
            },
        },
    )
    for _ in range(200):
        job = client.get(f"/api/jobs/{reply['job_id']}")
        if job["status"] not in {"queued", "running"}:
            break
        time.sleep(0.01)
    regions = service.store.list(RegionProposal, project["id"])
    if outcome == "empty":
        assert job["status"] == "succeeded"
        assert len(calls) == 11
        assert len(regions) == 1 and regions[0].bounds == []
    else:
        assert job["status"] == outcome
        assert len(calls) == 2
        assert regions == []
