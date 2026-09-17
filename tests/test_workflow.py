import base64

import nibabel as nib
import numpy as np

from monailabel.client.demo import run_demo
from monailabel.core.models import Asset, ModelRecord, Snapshot


def test_full_learning_and_handoff(client, http):
    result = run_demo(client, report=lambda _: None)
    evaluation = client.get(f"/api/evaluations/{result['evaluation_id']}")
    assert evaluation["candidate"]["mean_dice"] > 0.98
    assert evaluation["baseline"]["mean_dice"] < 0.6
    assert result["second_training_samples"] == 1
    assert evaluation["eligible_labels"] == [1, 2]
    snapshot = http.app.state.services.store.get(Snapshot, result["snapshot_id"])
    assert len(snapshot.samples) == 9
    assert all(s.decision_id for s in snapshot.samples)
    train_groups = {s.group_id for s in snapshot.samples if s.split == "train"}
    validation_groups = {s.group_id for s in snapshot.samples if s.split == "validation"}
    assert train_groups.isdisjoint(validation_groups)
    candidate = http.app.state.services.store.get(ModelRecord, result["model_id"])
    assert set(candidate.training_groups) == train_groups
    project = client.post(f"/api/promotions/{result['promotion_id']}/rollback")
    assert set(project["defaults"].values()) == {result["baseline_id"]}


def test_revisions_decisions_and_snapshot_eligibility(client, http, seeded):
    setup, assets = seeded
    asset = assets[0]
    prefix = f"/api/projects/{setup['project_id']}"
    proposal = client.wait(client.post(f"/api/assets/{asset['id']}/annotate", {})["id"])
    body = {
        "base_revision": 0,
        "proposal_id": proposal["proposal_id"],
        "covered_labels": [0, 1, 2],
        "reviewer": "forged-name",
    }
    annotation = client.post(f"/api/assets/{asset['id']}/review", body)
    assert annotation["reviewer"] == client.get("/api/auth/me")["id"]
    assert http.post(prefix + "/snapshots").status_code == 422
    rejected = client.post(
        f"/api/annotations/{annotation['id']}/decision",
        {"verdict": "changes_requested", "comment": "Boundary needs correction"},
    )
    assert rejected["reviewer_id"] == annotation["reviewer"]
    assert http.post(prefix + "/snapshots").status_code == 422
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    snapshot = client.post(prefix + "/snapshots")
    assert snapshot["samples"][0]["revision"] == 1
    assert http.post(f"/api/assets/{asset['id']}/review", json=body).status_code == 409
    # Editing accepted work starts a new review cycle; the old snapshot is immutable.
    reference = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
    new = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 1, "mask": reference, "covered_labels": [0, 1, 2]},
    )
    assert new["revision"] == 2
    assert http.post(prefix + "/snapshots").status_code == 422
    assert (
        http.post(
            f"/api/annotations/{annotation['id']}/decision", json={"verdict": "accepted"}
        ).status_code
        == 409
    )
    saved = http.app.state.services.store.get(Snapshot, snapshot["id"])
    assert saved.samples[0].revision == 1
    restored = client.post(
        f"/api/assets/{asset['id']}/restore",
        {"annotation_id": annotation["id"], "base_revision": 2, "reviewer": "ignored"},
    )
    assert restored["revision"] == 3
    assert restored["restored_from"] == annotation["id"]
    assert http.post(prefix + "/snapshots").status_code == 422


def test_nifti_geometry_and_slice_scope(client, http, seeded):
    setup, assets = seeded
    asset = assets[0]
    source = http.get(f"/api/assets/{asset['id']}/image").content
    volume = nib.Nifti1Image.from_bytes(source)
    np.testing.assert_allclose(volume.affine, asset["affine"])
    shape = asset["spatial_shape"]
    original = np.zeros(shape, dtype=np.uint8)
    original[0, 0, 0] = 2
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": original.tolist(), "covered_labels": [0, 1, 2]},
    )
    job = client.post(
        f"/api/assets/{asset['id']}/annotate", {"slice": {"axis": 2, "index": 12}, "label_ids": [1]}
    )
    result = client.wait(job["id"])
    mask = np.array(client.get(f"/api/proposals/{result['proposal_id']}/mask")["mask"])
    assert mask[0, 0, 0] == 2
    assert np.count_nonzero(mask[:, :, 12]) > 0
    np.testing.assert_array_equal(mask[:, :, :12], original[:, :, :12])
    np.testing.assert_array_equal(mask[:, :, 13:], original[:, :, 13:])
    assert (
        http.post(
            f"/api/assets/{asset['id']}/annotate", json={"slice": {"axis": 2, "index": 100}}
        ).status_code
        == 422
    )
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 1, "proposal_id": result["proposal_id"], "covered_labels": [0, 1, 2]},
    )
    exported = nib.Nifti1Image.from_bytes(
        http.get(f"/api/assets/{asset['id']}/segmentation.nii").content
    )
    np.testing.assert_allclose(exported.affine, volume.affine)
    np.testing.assert_array_equal(np.asarray(exported.dataobj), mask)
    # Source groups may not cross train/validation even when files differ.
    response = http.post(
        f"/api/projects/{setup['project_id']}/assets",
        json={
            "name": "copy.nii",
            "group_id": asset["group_id"],
            "split": "validation",
            "image_base64": base64.b64encode(source).decode(),
        },
    )
    assert response.status_code == 409


def test_snapshot_excludes_partial_labels(client, http, seeded):
    setup, assets = seeded
    asset = assets[0]
    proposal = client.wait(client.post(f"/api/assets/{asset['id']}/annotate", {})["id"])
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "proposal_id": proposal["proposal_id"], "covered_labels": [1]},
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    assert http.post(f"/api/projects/{setup['project_id']}/snapshots").status_code == 422


def test_validation_lineage_cannot_leak(client, http):
    result = run_demo(client, report=lambda _: None)
    service = http.app.state.services
    model = service.store.get(ModelRecord, result["model_id"])
    validation = next(
        a for a in service.store.list(Asset, result["project_id"]) if a.split == "validation"
    )
    with service.store.transaction() as session:
        session.update(
            model.model_copy(
                update={"training_groups": model.training_groups + [validation.group_id]}
            )
        )
    response = http.post(
        f"/api/projects/{result['project_id']}/evaluate",
        json={
            "snapshot_id": result["snapshot_id"],
            "candidate_id": model.id,
            "baseline_id": result["baseline_id"],
        },
    )
    assert response.status_code == 422
    assert "lineage" in response.text


def test_slice_provider_receives_display_grid_and_mask_returns_to_source(client, http, monkeypatch):
    import httpx

    from monailabel.server.data import nifti_bytes

    project = client.post(
        "/api/projects",
        {
            "name": "Display mapping",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Target", "color": "#ff0000"},
            ],
        },
    )
    image = np.zeros((2, 3, 4), dtype=np.float32)
    image[0, 1, 2] = 42
    asset = client.post(
        f"/api/projects/{project['id']}/assets",
        {
            "name": "asymmetric.nii",
            "group_id": "g",
            "split": "pool",
            "image_base64": base64.b64encode(nifti_bytes(image, np.eye(4).tolist())).decode(),
        },
    )
    model = client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "Display endpoint",
            "provider": "http-mask",
            "label_ids": [0, 1],
            "config": {"url": "https://test.example/predict"},
        },
    )

    def handler(request):
        import json

        payload = json.loads(request.content)
        assert payload["spatial_shape"] == [3, 2]
        assert payload["image"][1][1] == [42.0]
        return httpx.Response(200, json={"mask": [[0, 0], [0, 1], [0, 0]]})

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    job = client.post(
        f"/api/assets/{asset['id']}/annotate",
        {
            "model_id": model["id"],
            "slice": {
                "axis": 2,
                "index": 2,
                "orientation": {
                    "transpose": True,
                    "flip_rows": True,
                    "flip_columns": True,
                },
            },
        },
    )
    proposal = client.wait(job["id"])
    output = np.asarray(client.get(f"/api/proposals/{proposal['proposal_id']}/mask")["mask"])
    assert output.shape == (2, 3, 4)
    assert np.count_nonzero(output) == 1
    assert output[0, 1, 2] == 1
