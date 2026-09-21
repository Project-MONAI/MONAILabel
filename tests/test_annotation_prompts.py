import gzip
import json
import threading

import httpx
import nibabel as nib
import numpy as np
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import Proposal
from monailabel.core.ports import Prediction


@pytest.fixture
def annotation_project(client, http):
    project = client.post(
        "/api/projects",
        {
            "name": "Prompt scopes",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Spleen", "color": "#ff0000"},
                {"id": 2, "name": "Kidney", "color": "#00ff00"},
            ],
        },
    )
    image = np.broadcast_to(np.arange(3, dtype=np.float32), (4, 5, 3)).copy()
    response = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "volume.nii.gz", "group_id": "case"},
        content=gzip.compress(nib.Nifti1Image(image, np.eye(4)).to_bytes()),
    )
    assert response.status_code == 201
    return project, response.json()


def model_and_context(client, project, asset):
    model = client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "2D test model",
            "provider": "openai-chat-polygons",
            "label_ids": [0, 1],
            "config": {"url": "https://example.test/chat/completions", "model": "test"},
        },
    )
    return {
        "asset_id": asset["id"],
        "model_id": model["id"],
        "slice": {
            "axis": 2,
            "index": 1,
            "window": [0, 2],
            "orientation": {"transpose": True, "flip_rows": True, "flip_columns": False},
        },
    }


def test_missing_model_explains_setup_without_launching_job(client, annotation_project):
    project, asset = annotation_project
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": "annotate this slice", "context": {"asset_id": asset["id"]}},
    )
    assert reply["job_id"] is None
    assert reply["data"] == {"form": "model"}
    assert "No annotation model" in reply["message"]
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.mark.parametrize(
    ("message", "expected_calls", "all_slices"),
    [
        ("annotate this slice for spleen", 1, False),
        ("annotate all slices for spleen", 3, True),
        ("segment spleen in every slice", 3, True),
        ("segment spleen in this volume", 3, True),
    ],
)
def test_single_and_all_slice_prompts_preserve_geometry_and_other_labels(
    client, http, annotation_project, message, expected_calls, all_slices
):
    project, asset = annotation_project
    context = model_and_context(client, project, asset)
    previous = np.zeros(asset["spatial_shape"], dtype=np.uint8)
    previous[3, 0, 0] = 1
    previous[3, 3, :] = 2
    submitted = http.post(
        f"/api/assets/{asset['id']}/review-mask",
        params={"base_revision": 0, "covered_labels": [0, 1, 2]},
        content=previous.tobytes(),
    )
    assert submitted.status_code == 201
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            assert image.shape == (5, 4, 1)
            calls.append(float(image[0, 0, 0]))
            mask = np.zeros(image.shape[:-1], dtype=np.uint8)
            mask[0, 0] = 1
            return Prediction(mask)

    http.app.state.services.models.providers["openai-chat-polygons"] = Segmenter()
    reply = client.post(
        f"/api/projects/{project['id']}/assistant", {"message": message, "context": context}
    )
    result = client.wait(reply["job_id"])
    assert len(calls) == expected_calls
    assert calls == ([0, 0.5, 1] if all_slices else [0.5])
    proposal = client.get(f"/api/proposals/{result['proposal_id']}")
    assert proposal["all_slices"] is all_slices
    if all_slices:
        assert proposal["slice"] is None  # Older clients must also merge the full volume.
        assert proposal["volume_plane"] == context["slice"]
    assert proposal["base_revision"] == 1
    binary = http.get(f"/api/proposals/{proposal['id']}/mask.bin").content
    mask = np.frombuffer(binary, dtype=np.uint8).reshape(asset["spatial_shape"])
    np.testing.assert_array_equal(mask[3, 3, :], [2, 2, 2])
    if all_slices:
        np.testing.assert_array_equal(mask[0, 4, :], [1, 1, 1])
        assert mask[3, 0, 0] == 0
    else:
        assert mask[0, 4, 1] == 1
        np.testing.assert_array_equal(mask[:, :, [0, 2]], previous[:, :, [0, 2]])
    job = client.get(f"/api/jobs/{reply['job_id']}")
    assert f"of {expected_calls}" in job["progress_message"]
    # A proposal alone never advances the annotation revision or reviewer acceptance.
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1


def test_cancel_all_slices_stops_further_requests_and_publishes_no_proposal(
    client, http, annotation_project
):
    project, asset = annotation_project
    context = model_and_context(client, project, asset)
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append(1)
            entered.set()
            assert release.wait(10)
            return Prediction(np.ones(image.shape[:-1], dtype=np.uint8))

    http.app.state.services.models.providers["openai-chat-polygons"] = Segmenter()
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": "annotate all slices for spleen", "context": context},
    )
    try:
        assert entered.wait(5)
        assert "Slice 1 of 3" in client.get(f"/api/jobs/{reply['job_id']}")["progress_message"]
        client.post(f"/api/jobs/{reply['job_id']}/cancel")
    finally:
        release.set()
        http.app.state.services.jobs.executor.shutdown(wait=True)
    assert len(calls) == 1
    assert http.app.state.services.store.list(Proposal, project["id"]) == []
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0


def test_binary_review_checks_size_labels_and_revision(client, http, annotation_project):
    project, asset = annotation_project
    path = f"/api/assets/{asset['id']}/review-mask"
    params = {"base_revision": 0, "covered_labels": [0, 1, 2]}
    assert http.post(path, params=params, content=b"x").status_code == 422
    mask = np.zeros(asset["spatial_shape"], dtype=np.uint8)
    mask[1, 2, 1] = 9
    assert http.post(path, params=params, content=mask.tobytes()).status_code == 422
    mask[1, 2, 1] = 1
    response = http.post(path, params=params, content=mask.tobytes())
    assert response.status_code == 201
    saved = response.json()
    assert http.get(f"/api/annotations/{saved['id']}/mask.bin").content == mask.tobytes()
    assert http.post(path, params=params, content=mask.tobytes()).status_code == 409
    assert http.post(f"/api/projects/{project['id']}/snapshots").status_code == 422


@pytest.mark.parametrize("selected", [None, "sol"])
@pytest.mark.parametrize("named", ["astra", "claude"])
def test_prompt_named_model_overrides_selection_and_default_through_provider_http(
    client, http, annotation_project, monkeypatch, selected, named
):
    project, asset = annotation_project
    registered = {}
    for name, title, provider_id in [
        ("sol", "GPT-5.6 Sol", "switchyard/openai/gpt-5.6-sol"),
        ("astra", "GPT-6 Astra", "azure/openai/gpt-6-astra"),
        ("claude", "Claude Opus 5", "azure/anthropic/claude-opus-5"),
    ]:
        registered[name] = client.post(
            f"/api/projects/{project['id']}/models",
            {
                "name": title,
                "provider": "openai-chat-polygons",
                "label_ids": [0, 1],
                "config": {"url": "https://example.test/chat/completions", "model": provider_id},
            },
        )
    default = client.request(
        "PUT",
        f"/api/projects/{project['id']}/defaults",
        {"model_id": registered["sol"]["id"], "label_ids": [1], "base_version": 0},
    )
    calls = []

    def predict(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps({"polygons": []})}}
                ]
            },
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(predict), **kwargs),
    )
    context = {
        "asset_id": asset["id"],
        "label_ids": [1],
        "slice": {"axis": 2, "index": 1, "window": [0, 2]},
    }
    if selected:
        context["model_id"] = registered[selected]["id"]
    chosen = registered[named]
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="named-model",
                    name="annotate",
                    arguments={"model_name": chosen["name"], "scope": "current_slice"},
                )
            ],
        )
    ]
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": f"annotate this slice using {chosen['name']}", "context": context},
    )
    result = client.wait(reply["job_id"])
    assert chosen["name"] in reply["message"] and "Sol" not in reply["message"]
    assert client.get(f"/api/jobs/{reply['job_id']}")["request"]["model_id"] == chosen["id"]
    assert client.get(f"/api/proposals/{result['proposal_id']}")["model_ids"] == [chosen["id"]]
    # Prompt-specific overrides are not saved as project defaults or sticky chat context.
    assert "model_id" not in reply["data"]
    assert client.get(f"/api/projects/{project['id']}")["defaults"] == default["defaults"]
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": "annotate this slice", "context": context},
    )
    client.wait(reply["job_id"])
    assert calls == [chosen["config"]["model"], "switchyard/openai/gpt-5.6-sol"]
    count = len(client.get(f"/api/projects/{project['id']}/jobs"))
    for message in ["using Missing model", "using GPT model"]:
        response = http.post(
            f"/api/projects/{project['id']}/assistant",
            json={"message": f"annotate this slice {message}", "context": context},
        )
        assert response.status_code == 422
    assert len(client.get(f"/api/projects/{project['id']}/jobs")) == count
    assert len(calls) == 2


@pytest.mark.parametrize("all_slices", [False, True])
def test_native_3d_model_runs_once_and_preserves_the_requested_edit_scope(
    client, http, annotation_project, all_slices, monkeypatch
):
    from monailabel.core.models import ModelRecord

    project, asset = annotation_project
    service = http.app.state.services
    model = ModelRecord(
        project_id=project["id"], name="Native network", provider="monai-unet", label_ids=[0, 1]
    )
    with service.store.transaction() as session:
        session.insert(model)
    calls = []

    def predict(project, model, image, prompt, affine=None):
        assert image.shape == (4, 5, 3, 1)
        assert affine == asset["affine"]
        calls.append(True)
        mask = np.zeros((4, 5, 3), dtype=np.uint8)
        mask[1, 2, :] = 1
        return mask

    monkeypatch.setattr(service.models, "predict", predict)
    job = client.post(
        f"/api/assets/{asset['id']}/annotate",
        {
            "model_id": model.id,
            "label_ids": [1],
            "all_slices": all_slices,
            "slice": {"axis": 2, "index": 1, "orientation": {"transpose": True, "flip_rows": True}},
        },
    )
    result = client.wait(job["id"])
    mask = np.frombuffer(
        http.get(f"/api/proposals/{result['proposal_id']}/mask.bin").content, dtype=np.uint8
    ).reshape((4, 5, 3))
    assert len(calls) == 1
    assert mask[1, 2, :].tolist() == ([1, 1, 1] if all_slices else [0, 1, 0])
    assert mask.sum() == (3 if all_slices else 1)
