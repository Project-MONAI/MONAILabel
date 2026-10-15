# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All-structure selection uses the pinned catalog, not chat or project defaults."""

import gzip
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, Project
from monailabel.core.ports import Prediction, Volume
from monailabel.totalsegmentator import prepared
from monailabel.totalsegmentator.catalog import targets
from monailabel.totalsegmentator.config import TotalConfig
from monailabel.totalsegmentator.runtime import TotalSegmenter


@pytest.fixture
def all_structures(client, http, monkeypatch):
    service = http.app.state.services
    service.presets.enabled, service.presets.hosted = True, {}
    project = client.post(
        "/api/projects",
        {
            "name": "All structures",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 7, "name": "Spleen"},
                {"id": 17, "name": "Kidney left"},
            ],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    models = client.get(prefix + "/models")
    image = np.zeros((117, 3, 2), dtype=np.int16)
    affine = np.diag([-1.7, 2.1, 3.8, 1.0])
    asset = http.post(
        prefix + "/assets/upload",
        params={"name": "ct.nii.gz"},
        content=gzip.compress(nib.Nifti1Image(image, affine).to_bytes()),
    ).json()
    calls = []

    class Segmenter:
        def predict_volume(self, volume, labels, prompt, model):
            calls.append(labels)
            np.testing.assert_allclose(volume.affine, affine)
            mask = np.zeros(image.shape, np.uint8)
            for index, label in enumerate(label for label in labels if label.id):
                mask[index] = label.id
            return Prediction(mask)

    monkeypatch.setattr(service.models.recipes, "segmenter", lambda *args: Segmenter())
    model = next(m for m in models if m["provider"] == "totalsegmentator-ct")
    context = {
        "asset_id": asset["id"],
        "base_revision": 0,
        "model_id": model["id"],
        "slice": {"axis": 2, "index": 0},
        "interaction_target": "",
    }
    return prefix, context, calls


def test_direct_all_catalog_then_binary_review_preserves_ids(client, http, all_structures):
    prefix, context, calls = all_structures
    reply = client.post(
        prefix + "/viewer-inference", {"context": context, "scope": "full", "all_targets": True}
    )
    result = client.wait(reply["job_id"])
    labels = reply["data"]["project"]["labels"]
    assert len(labels) == 118
    assert next(x["id"] for x in labels if x["name"] == "Spleen") == 7
    assert next(x["id"] for x in labels if x["name"] == "Kidney left") == 17
    assert len(calls) == 1 and len(calls[0]) == 118
    foreground = {x["id"] for x in labels if x["id"]}
    assert max(foreground) > 127
    proposal = client.get("/api/proposals/" + result["proposal_id"])
    assert set(proposal["label_ids"]) == foreground and proposal["slice"] is None
    binary = http.get(f"/api/proposals/{proposal['id']}/mask.bin").content
    assert set(np.frombuffer(binary, np.uint8)) == foreground
    response = http.post(
        f"/api/assets/{context['asset_id']}/review-mask",
        params=[("base_revision", 0), *[("covered_labels", i) for i in [0, *sorted(foreground)]]],
        content=binary,
    )
    assert response.status_code == 201, response.text
    assert len(response.json()["covered_labels"]) == 118
    assert http.app.state.services.store.get(Project, prefix.rsplit("/", 1)[-1]).labels


@pytest.mark.parametrize(
    "scope,change",
    [("current_slice", {}), ("full", {"base_revision": 99}), ("full", {"model_id": "foreign"})],
)
def test_invalid_all_requests_never_start_inference(http, all_structures, scope, change):
    prefix, context, calls = all_structures
    response = http.post(
        prefix + "/viewer-inference",
        json={"context": context | change, "scope": scope, "all_targets": True},
    )
    assert response.status_code in {404, 409, 422}
    assert calls == []


def test_single_target_still_requests_one_structure(client, all_structures):
    prefix, context, calls = all_structures
    reply = client.post(
        prefix + "/viewer-inference",
        {"context": context | {"interaction_target": "Liver"}, "scope": "full"},
    )
    client.wait(reply["job_id"])
    assert [x.name.casefold() for x in calls[0] if x.id] == ["liver"]


@pytest.mark.parametrize("fail", [False, True])
def test_combined_replacement_never_clears_saved_work_first(
    client, http, all_structures, monkeypatch, fail
):
    prefix, context, calls = all_structures
    service = http.app.state.services
    original = np.full((117, 3, 2), 7, np.uint8)
    annotation = client.post(
        f"/api/assets/{context['asset_id']}/review",
        {"base_revision": 0, "mask": original.tolist(), "covered_labels": [0, 7, 17]},
    )
    saved = http.get(f"/api/annotations/{annotation['id']}/mask.bin").content

    def broken(*args):
        raise DomainError("Test inference failure")

    if fail:
        monkeypatch.setattr(service.models.recipes, "segmenter", broken)
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="replace",
                    name="annotate",
                    arguments={
                        "model_name": "TotalSegmentator CT",
                        "all_targets": True,
                        "replace_existing": True,
                        "scope": "full",
                    },
                )
            ],
        )
    ]
    reply = client.post(
        prefix + "/assistant",
        {
            "message": "delete the current segmentation and run TotalSegmentator CT on all labels",
            "context": context | {"base_revision": 1, "viewer_actions": ["clear_segments"]},
        },
    )
    assert reply.get("job_id") and reply["data"].get("client_action") != "clear_segments"
    if fail:
        with pytest.raises(Exception, match="Test inference failure"):
            client.wait(reply["job_id"])
    else:
        result = client.wait(reply["job_id"])
        proposal = client.get("/api/proposals/" + result["proposal_id"])
        assert proposal["replace_existing"] is True
        assert proposal["base_revision"] == 1 and len(proposal["label_ids"]) == 117
    assert http.get(f"/api/annotations/{annotation['id']}/mask.bin").content == saved
    assert client.get("/api/assets/" + context["asset_id"])["annotation_id"] == annotation["id"]


def test_prepared_prediction_is_read_only_geometry_keyed_and_remapped(tmp_path, monkeypatch):
    volume = Volume(np.zeros((2, 3, 4, 1), np.float32), np.eye(4).tolist())
    provider, config = "totalsegmentator-ct", TotalConfig()
    key = prepared.identity(provider, volume, config)
    folder = tmp_path / key
    folder.mkdir()
    np.save(folder / "mask.npy", np.full((2, 3, 4), targets(provider)["liver"], np.uint8))
    checksum = hashlib.sha256((folder / "mask.npy").read_bytes()).hexdigest()
    (folder / "prediction.json").write_text(json.dumps({"identity": key, "mask_sha256": checksum}))
    monkeypatch.setenv("MONAILABEL_TOTALSEG_PREDICTIONS", str(tmp_path))

    def never(*args):
        raise AssertionError("Cache hit must not verify/download weights or invoke inference")

    monkeypatch.setattr("monailabel.totalsegmentator.runtime.pretrained_weights", never)
    model = ModelRecord(name="CT", provider=provider, label_ids=[0, 240])
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = TotalSegmenter(provider, {}, None).predict_volume(
        volume, [Label(id=240, name="Liver")], "", model
    )
    assert np.all(result.mask == 240)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    moved = Volume(volume.image, np.diag([2, 1, 1, 1]).astype(float).tolist())
    assert prepared.identity(provider, moved, config) != key
    with pytest.raises(AssertionError, match="Cache hit"):
        TotalSegmenter(provider, {"derived": True}, None).predict_volume(
            volume, [Label(id=1, name="Liver")], "", model
        )
    (folder / "mask.npy").write_bytes(b"corrupt")
    with pytest.raises(DomainError, match="invalid"):
        TotalSegmenter(provider, {}, None).predict_volume(
            volume, [Label(id=1, name="Liver")], "", model
        )
