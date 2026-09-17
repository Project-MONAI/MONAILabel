"""Spatial annotation contracts, scope isolation and provider selection."""

import io

import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    AnnotateRequest,
    Asset,
    ImageRegion,
    ModelRecord,
    PromptPoint,
    SliceScope,
    SpatialPrompt,
)
from monailabel.core.ports import Prediction
from monailabel.server import spatial_annotation
from monailabel.server.models import Models


def asset(shape=(6, 7, 4)):
    return Asset(
        project_id="p",
        name="sample",
        group_id="g",
        split="pool",
        kind="volume3d" if len(shape) == 3 else "image2d",
        spatial_shape=list(shape),
        image_key="image",
        source_key="source",
    )


def model(provider="medsam2"):
    return ModelRecord(
        project_id="p", name=provider, provider=provider, label_ids=[0], read_only=True
    )


def test_spatial_contract_rejects_unsafe_and_ambiguous_geometry():
    for value in [
        {},
        {"box": [[1, 2], [0, 4]]},
        {"box": [[1, 2], [3, 4, 5]]},
        {"points": [{"coordinates": [1, float("nan")]}]},
        {"points": [{"coordinates": [1, 2], "positive": False}]},
    ]:
        with pytest.raises(ValueError):
            SpatialPrompt.model_validate(value)
    assert SpatialPrompt(
        points=[PromptPoint(coordinates=[2, 3]), PromptPoint(coordinates=[4, 5], positive=False)]
    )


def test_sam_requires_valid_source_hints_and_one_target():
    plane = SliceScope(axis=2, index=2, window=[0, 1])
    request = AnnotateRequest(slice=plane, spatial_prompt=SpatialPrompt(box=[[1, 2, 2], [4, 5, 2]]))
    assert spatial_annotation.validate(asset(), request, model(), [1]) == request.spatial_prompt
    for body, selected, labels in [
        (request.model_copy(update={"spatial_prompt": None}), model(), [1]),
        (request.model_copy(update={"slice": None}), model(), [1]),
        (request, model(), [1, 2]),
        (request.model_copy(update={"all_slices": True}), model("sam2"), [1]),
        (
            request.model_copy(
                update={"spatial_prompt": SpatialPrompt(box=[[1, 2, 2], [6, 5, 2]])}
            ),
            model(),
            [1],
        ),
        (
            request.model_copy(
                update={
                    "spatial_prompt": SpatialPrompt(points=[PromptPoint(coordinates=[2, 3, 1])])
                }
            ),
            model(),
            [1],
        ),
    ]:
        with pytest.raises(DomainError):
            spatial_annotation.validate(asset(), body, selected, labels)
    assert Models.promptable(model()) and Models.requires_spatial(model())
    Models.validate_targets(model(), ["New organ"])


class Predictor:
    def __init__(self):
        self.calls = []

    def predict_prompted(self, image, label, model, spatial, plane, full, progress):
        self.calls.append((image.shape, spatial, plane, full))
        progress(0.5)
        return Prediction(np.full(image.shape[:-1], label, dtype=np.uint8))


def test_spatial_slice_preserves_other_slices_and_overlapping_labels():
    image = np.zeros((6, 7, 4, 1), dtype=np.float32)
    original = np.zeros(image.shape[:-1], dtype=np.uint8)
    original[1, 1, 0] = 2
    req = AnnotateRequest(
        slice=SliceScope(axis=2, index=2, window=[0, 1]),
        spatial_prompt=SpatialPrompt(box=[[1, 2, 2], [4, 5, 2]]),
    )
    hints = spatial_annotation.validate(asset(), req, model(), [1])
    result = spatial_annotation.predict(
        Predictor(), image, original.copy(), model(), req, hints, 1, lambda p: None
    )
    assert (result[:, :, 2] == 1).all()
    np.testing.assert_array_equal(result[:, :, [0, 1, 3]], original[:, :, [0, 1, 3]])
    original[1, 1, 2] = 2
    with pytest.raises(Conflict):
        spatial_annotation.predict(
            Predictor(), image, original.copy(), model(), req, hints, 1, lambda p: None
        )


def test_selected_pathology_region_is_one_crop_with_source_restoration():
    image = np.zeros((20, 30, 3), dtype=np.float32)
    original = np.zeros((20, 30), dtype=np.uint8)
    original[0, 0] = 2
    crop = ImageRegion(x=4, y=3, width=7, height=5, runs=[(0, 10)])
    req = AnnotateRequest(image_region=crop)
    hints = spatial_annotation.validate(asset((20, 30)), req, model("sam2"), [1])
    provider = Predictor()
    result = spatial_annotation.predict(
        provider, image, original.copy(), model("sam2"), req, hints, 1, lambda p: None
    )
    assert provider.calls == [((5, 7, 3), SpatialPrompt(box=[[0, 0], [4, 6]]), None, False)]
    assert np.count_nonzero(result == 1) == 10 and result[0, 0] == 2


def test_sam_job_records_hints_and_missing_hints_never_call_provider(client, http, monkeypatch):
    service = http.app.state.services
    service.presets.enabled = True
    project = client.post(
        "/api/projects",
        {
            "name": "SAM workflow",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Object", "color": "#00ff00"},
            ],
        },
    )
    content = io.BytesIO()
    Image.fromarray(np.zeros((6, 7), dtype=np.uint8)).save(content, format="PNG")
    response = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "test.png"},
        content=content.getvalue(),
    )
    assert response.status_code == 201
    sample = response.json()
    sam = next(
        m for m in client.get(f"/api/projects/{project['id']}/models") if m["provider"] == "sam2"
    )
    provider = Predictor()
    monkeypatch.setattr(service.models, "spatial_provider", lambda: provider)
    path = "/api/assets/" + sample["id"] + "/annotate"
    rejected = http.post(path, json={"model_id": sam["id"], "label_ids": [1]})
    assert rejected.status_code == 422 and "spatial hint" in rejected.text
    assert provider.calls == []
    hints = {"box": [[1, 2], [4, 5]], "points": []}
    result = client.wait(
        client.post(path, {"model_id": sam["id"], "label_ids": [1], "spatial_prompt": hints})["id"]
    )
    proposal = client.get("/api/proposals/" + result["proposal_id"])
    assert proposal["spatial_prompt"] == hints and proposal["base_revision"] == 0
    assert proposal["model_ids"] == [sam["id"]]
