import gzip

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Project, Snapshot
from monailabel.core.ports import Prediction
from monailabel.server.labels import resolve_labels


def test_empty_project_adds_targets_without_changing_existing_masks_or_models(client, http):
    project = client.post("/api/projects", {"name": "Radiology"})
    assert [x["id"] for x in project["labels"]] == [0]
    prefix = f"/api/projects/{project['id']}"
    model = client.post(
        prefix + "/models",
        {
            "name": "Promptable vision",
            "provider": "openai-chat-polygons",
            "config": {"url": "https://unused.test/chat/completions", "model": "fixture"},
        },
    )
    source = np.zeros((4, 5, 3), dtype=np.float32)
    asset = http.post(
        prefix + "/assets/upload",
        params={"name": "scan.nii.gz", "group_id": "patient"},
        content=gzip.compress(nib.Nifti1Image(source, np.eye(4)).to_bytes()),
    ).json()
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(([label.name for label in labels], model.label_ids))
            mask = np.zeros(image.shape[:-1], dtype=np.uint8)
            mask[1, model.label_ids[-1]] = model.label_ids[-1]
            return Prediction(mask)

    http.app.state.services.models.providers[model["provider"]] = Segmenter()
    context = {"asset_id": asset["id"], "slice": {"axis": 2, "index": 1, "window": [0, 1]}}
    reply = client.post(
        prefix + "/assistant", {"message": "annotate spleen on this slice", "context": context}
    )
    first = client.wait(reply["job_id"])
    updated = reply["data"]["project"]
    assert [(label["id"], label["name"]) for label in updated["labels"]] == [
        (0, "Background"),
        (1, "Spleen"),
    ]
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "proposal_id": first["proposal_id"], "covered_labels": [0, 1]},
    )
    original = http.get(f"/api/annotations/{annotation['id']}/mask.bin").content
    reply = client.post(
        prefix + "/assistant", {"message": "annotate the liver on this slice", "context": context}
    )
    second = client.wait(reply["job_id"])
    mask = np.frombuffer(
        http.get(f"/api/proposals/{second['proposal_id']}/mask.bin").content, np.uint8
    ).reshape(source.shape)
    assert mask[1, 1, 1] == 1 and mask[1, 2, 1] == 2
    assert calls == [(["Background", "Spleen"], [0, 1]), (["Background", "Liver"], [0, 2])]
    assert http.get(f"/api/annotations/{annotation['id']}/mask.bin").content == original
    current = client.get(prefix)
    assert [label["id"] for label in current["labels"]] == [0, 1, 2]
    assert current["protocol_version"] == project["protocol_version"]
    assert current["defaults"] == {"1": model["id"], "2": model["id"]}
    assert client.get(prefix + "/models")[0]["label_ids"] == [0]
    service = http.app.state.services
    reused, ids = resolve_labels(service.store, project["id"], ["SPLEEN"], model["id"])
    assert ids == [1] and reused.version == current["version"]
    assert (
        client.request(
            "PUT",
            prefix + "/defaults",
            {"model_id": model["id"], "label_ids": [2], "base_version": current["version"]},
        )["defaults"]["2"]
        == model["id"]
    )


def test_fixed_model_cannot_silently_add_an_untrained_target(client, http, seeded):
    setup, _ = seeded
    store = http.app.state.services.store
    project = store.get(Project, setup["project_id"])
    before = project.model_dump()
    with pytest.raises(DomainError, match="fixed targets"):
        resolve_labels(store, project.id, ["liver"], next(iter(project.defaults.values())))
    assert store.get(Project, project.id).model_dump() == before


def test_label_scoped_snapshot_keeps_existing_accepted_work_eligible(client, http, seeded):
    setup, assets = seeded
    service = http.app.state.services
    project_id = setup["project_id"]
    asset = next(a for a in assets if a["split"] == "train")
    reference = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": reference, "covered_labels": [0, 1, 2]},
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    old = service.datasets.snapshot(project_id)
    model = client.post(
        f"/api/projects/{project_id}/models",
        {
            "name": "Vision",
            "provider": "openai-chat-polygons",
            "config": {"url": "https://unused.test/chat/completions", "model": "fixture"},
        },
    )
    resolve_labels(service.store, project_id, ["liver"], model["id"])
    with pytest.raises(DomainError, match="complete training"):
        service.datasets.snapshot(project_id)
    scoped = service.datasets.snapshot(project_id, [0, 1, 2])
    assert scoped.samples == old.samples and scoped.labels == old.labels
    assert service.store.get(Snapshot, old.id) == old
