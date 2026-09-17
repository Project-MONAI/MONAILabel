import io
import json

import numpy as np
import pytest
from PIL import Image

from monailabel.core.models import Annotation, ModelRecord, Project
from monailabel.server.data import nifti_bytes


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def pair(
    http,
    pid,
    body=None,
    *,
    number=0,
    volume=False,
    bad_geometry=False,
    mask_value=7,
    endpoint="evaluation-imports",
):
    mask = np.zeros((8, 8, 8) if volume else (8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = mask_value
    affine = np.eye(4).tolist()
    source = (
        mask.astype(np.float32) + number if volume else np.full((8, 8, 3), 90 + number, np.uint8)
    )
    geometry = np.diag([2, 1, 1, 1]).tolist() if bad_geometry else affine
    image_content = nifti_bytes(source, affine) if volume else png(source)
    label_content = nifti_bytes(mask, geometry) if volume else png(mask)
    suffix = ".nii" if volume else ".png"
    request = {"evaluation_set_name": "External references", "labels": {7: "Spleen"}} | (body or {})
    return http.post(
        f"/api/projects/{pid}/{endpoint}",
        data={"metadata": json.dumps(request)},
        files={
            "image": (f"case-{number}{suffix}", image_content),
            "labels": (f"case-{number}_mask{suffix}", label_content),
        },
    )


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("reviewed", [False, True])
def test_import_reserves_all_cases_maps_labels_and_requires_explicit_review(
    client, http, volume, reviewed
):
    pid = client.post("/api/projects", {"name": "External truth"})["id"]
    first = pair(http, pid, {"reviewed": reviewed}, volume=volume)
    assert first.status_code == 201, first.text
    record = first.json()["evaluation_set"]
    request = {
        "evaluation_set_name": None,
        "evaluation_set_id": record["id"],
        "base_version": record["version"],
        "reviewed": reviewed,
    }
    second = pair(http, pid, request, number=1, volume=volume)
    assert second.status_code == 201, second.text
    result = second.json()
    prefix = f"/api/projects/{pid}"
    assets = client.get(prefix + "/assets")
    assert len(assets) == 2 and all(a["split"] == "validation" for a in assets)
    assert len(result["evaluation_set"]["member_groups"]) == 2
    assert result["evaluation_set"]["percentage"] == 100
    assert not result["evaluation_set"]["auto_update"]
    assert client.get(prefix)["labels"][1]["name"] == "Spleen"
    assert result["covered_labels"] == [0, 1]
    assert set(
        np.unique(client.get(f"/api/annotations/{result['annotation_id']}/mask")["mask"])
    ) == {0, 1}
    assert len(client.get(prefix + "/decisions")) == (2 if reviewed else 0)
    version = http.post(
        prefix + f"/evaluation-sets/{record['id']}/versions",
        json={"base_version": result["evaluation_set"]["version"], "label_ids": [0, 1]},
    )
    assert version.status_code == (200 if reviewed else 422), version.text
    if reviewed:
        assert len(version.json()["samples"]) == 2
    # A duplicate import under a new patient ID still cannot enter training.
    image = np.full((8, 8, 3), 91, np.uint8)
    if not volume:
        response = http.post(
            prefix + "/assets/upload",
            params={"name": "alias.png", "group_id": "other-patient", "split": "train"},
            content=png(image),
        )
        assert response.status_code == 409


def test_bad_geometry_and_unknown_labels_leave_no_project_changes(client, http):
    project = client.post("/api/projects", {"name": "Geometry"})
    for options in [{"volume": True, "bad_geometry": True}, {"mask_value": 8}]:
        response = pair(http, project["id"], **options)
        assert response.status_code == 422, response.text
        assert client.get(f"/api/projects/{project['id']}/assets") == []
        assert client.get(f"/api/projects/{project['id']}/evaluation-sets") == []
        assert client.get(f"/api/projects/{project['id']}") == project


def test_reimport_keeps_annotation_and_rejects_overwrite_and_stale_set(client, http):
    pid = client.post("/api/projects", {"name": "Retry"})["id"]
    first = pair(http, pid, {"reviewed": True}).json()
    record = first["evaluation_set"]
    body = {
        "evaluation_set_name": None,
        "evaluation_set_id": record["id"],
        "base_version": record["version"],
        "reviewed": True,
    }
    again = pair(http, pid, body)
    assert again.status_code == 201
    assert again.json() == first
    annotation = http.app.state.services.store.get(Annotation, first["annotation_id"])
    # Same image content, different mask: refuse to overwrite saved labels.
    response = pair(http, pid, body | {"labels": {8: "Liver"}}, mask_value=8)
    assert response.status_code == 409, response.text
    assert http.app.state.services.store.get(Annotation, annotation.id) == annotation
    assert [label["name"] for label in client.get(f"/api/projects/{pid}")["labels"]] == [
        "Background",
        "Spleen",
    ]
    response = pair(http, pid, body | {"base_version": 0}, number=2)
    assert response.status_code == 409
    assert len(client.get(f"/api/projects/{pid}/assets")) == 1


def test_training_lineage_blocks_reference_import_without_partial_records(client, http):
    pid = client.post("/api/projects", {"name": "No leakage"})["id"]
    model = ModelRecord(
        project_id=pid,
        name="Used patient",
        provider="pixel-gaussian",
        label_ids=[0, 1],
        training_groups=["patient-1"],
    )
    with http.app.state.services.store.transaction() as session:
        session.insert(model)
    before = http.app.state.services.store.get(Project, pid)
    response = pair(http, pid, {"group_id": "patient-1", "reviewed": True})
    assert response.status_code == 409, response.text
    assert client.get(f"/api/projects/{pid}/assets") == []
    assert client.get(f"/api/projects/{pid}/evaluation-sets") == []
    assert http.app.state.services.store.get(Project, pid) == before
    # A different patient ID cannot disguise an already-used decoded image.
    existing = http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": "training.png", "group_id": "patient-1", "split": "train"},
        content=png(np.full((8, 8, 3), 90, np.uint8)),
    )
    assert existing.status_code == 201
    response = pair(http, pid, {"group_id": "new-alias", "reviewed": True})
    assert response.status_code == 409
    assert client.get(f"/api/projects/{pid}/assets") == [existing.json()]
    assert client.get(f"/api/projects/{pid}/evaluation-sets") == []


def test_reference_import_requires_manager_and_scopes_existing_set(client, http):
    pid = client.post("/api/projects", {"name": "Private evaluation"})["id"]
    first = pair(http, pid).json()
    other = client.post("/api/projects", {"name": "Other project"})["id"]
    response = pair(
        http,
        other,
        {
            "evaluation_set_name": None,
            "evaluation_set_id": first["evaluation_set"]["id"],
            "base_version": 1,
        },
    )
    assert response.status_code == 422
    user = client.post(
        "/api/auth/users", {"username": "annotator", "password": "annotator-password"}
    )
    client.request(
        "PUT", f"/api/projects/{pid}/members", {"user_id": user["id"], "roles": ["annotator"]}
    )
    client.post("/api/auth/login", {"username": "annotator", "password": "annotator-password"})
    assert pair(http, pid, {"reviewed": True}).status_code == 403


@pytest.mark.parametrize("split", ["pool", "train"])
@pytest.mark.parametrize("reviewed", [False, True])
def test_general_label_import_and_idempotent_retry(client, http, split, reviewed):
    pid = client.post("/api/projects", {"name": "Labeled files"})["id"]
    body = {"evaluation_set_name": None, "split": split, "reviewed": reviewed}
    first = pair(http, pid, body, endpoint="label-imports")
    assert first.status_code == 201, first.text
    assert first.json()["evaluation_set"] is None
    assert pair(http, pid, body, endpoint="label-imports").json() == first.json()
    assets = client.get(f"/api/projects/{pid}/assets")
    assert len(assets) == 1 and assets[0]["split"] == split
    assert assets[0]["revision"] == 1 and assets[0]["annotation_id"]
    assert client.get(f"/api/projects/{pid}/evaluation-sets") == []
    assert len(client.get(f"/api/projects/{pid}/decisions")) == int(reviewed)
    rejected = pair(http, pid, body | {"labels": {7: "Liver"}}, endpoint="label-imports")
    assert rejected.status_code == 409
    assert client.get(f"/api/projects/{pid}/assets") == assets


def test_label_import_cannot_release_evaluation_or_mix_training_groups(client, http):
    pid = client.post("/api/projects", {"name": "Protected uses"})["id"]
    assert pair(http, pid).status_code == 201
    body = {"evaluation_set_name": None, "group_id": "alias"}
    for split in ["pool", "train"]:
        response = pair(http, pid, body | {"split": split}, endpoint="label-imports")
        assert response.status_code == 409, response.text
    # A new case can enter training, but related annotation cases cannot cross splits.
    body["group_id"] = "training-patient"
    assert (
        pair(http, pid, body | {"split": "train"}, number=1, endpoint="label-imports").status_code
        == 201
    )
    response = pair(http, pid, body, number=2, endpoint="label-imports")
    assert response.status_code == 409
    assert len(client.get(f"/api/projects/{pid}/assets")) == 2


def test_import_contract_requires_evaluation_set_only_for_evaluation(client, http):
    pid = client.post("/api/projects", {"name": "Import purpose"})["id"]
    assert pair(http, pid, endpoint="label-imports").status_code == 422
    response = pair(
        http, pid, {"split": "validation", "evaluation_set_name": None}, endpoint="label-imports"
    )
    assert response.status_code == 422
    assert pair(http, pid, {"split": "train"}).status_code == 422
    assert client.get(f"/api/projects/{pid}/assets") == []
