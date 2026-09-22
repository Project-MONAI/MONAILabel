import io

import numpy as np
import pytest
from PIL import Image

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, Label, ModelRecord, Project
from monailabel.core.ports import Prediction


@pytest.fixture
def pathology(client, http, monkeypatch):
    monkeypatch.setenv("TEST_ANNOTATION_KEY", "fixture-only")
    project = client.post(
        "/api/projects",
        {
            "name": "Pathology",
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": "Nuclei", "color": "#ff0000"},
            ],
        },
    )
    source = io.BytesIO()
    Image.new("RGB", (32, 24), (220, 170, 190)).save(source, format="PNG")
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "slide.png", "group_id": "slide"},
        content=source.getvalue(),
    ).json()
    service = http.app.state.services
    models = {}
    with service.store.transaction() as session:
        for key, name, provider in [
            ("vista3d", "VISTA3D", "vista3d"),
            ("nvidia-astra", "GPT-6 Astra", "openai-chat-polygons"),
            ("nvidia-claude-opus-5", "Claude Opus 5", "openai-chat-polygons"),
            ("custom-claude", "My Claude", "anthropic-polygons"),
        ]:
            models[key] = ModelRecord(
                project_id=project["id"],
                name=name,
                provider=provider,
                label_ids=[0],
                preset=key,
                read_only=True,
                config={
                    "url": "https://unused.test/chat/completions",
                    "model": "fixture",
                    "token_env": "TEST_ANNOTATION_KEY",
                }
                if provider != "vista3d"
                else {},
            )
            session.insert(models[key])
        current = session.get(Project, project["id"])
        session.update(current.model_copy(update={"annotation_model_id": models["vista3d"].id}))
    return project["id"], asset["id"], models


@pytest.mark.parametrize(
    "selected", [None, "nvidia-astra", "nvidia-claude-opus-5", "custom-claude"]
)
def test_nuclei_region_uses_compatible_model_and_preserves_saved_mask(
    client, http, pathology, selected
):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    calls = []

    class Segmenter:
        def predict(self, image, labels, prompt, model):
            calls.append((image.shape, model.id, [label.name for label in labels]))
            return Prediction(np.ones(image.shape[:-1], dtype=np.uint8))

    service.models.providers["openai-chat-polygons"] = Segmenter()
    service.models.providers["anthropic-polygons"] = Segmenter()
    arguments = {"targets": ["Nuclei"], "scope": "selected_region"}
    if selected:
        arguments["model_name"] = models[selected].name
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="nuclei", name="annotate", arguments=arguments)],
        )
    ]
    reply = client.post(
        f"/api/projects/{project_id}/assistant",
        {
            "message": "Annotate this region for nuclie",
            "context": {
                "asset_id": asset_id,
                "image_region": {"x": 4, "y": 5, "width": 8, "height": 6},
            },
        },
    )
    chosen = models[selected or "nvidia-astra"]
    assert chosen.name in reply["message"]
    result = client.wait(reply["job_id"])
    assert calls == [((6, 8, 3), chosen.id, ["Background", "Nuclei"])]
    mask = np.frombuffer(
        http.get(f"/api/proposals/{result['proposal_id']}/mask.bin").content, np.uint8
    ).reshape(24, 32)
    assert mask[5:11, 4:12].sum() == 48 and mask.sum() == 48
    assert client.get(f"/api/assets/{asset_id}")["revision"] == 0
    metadata = service.assistants.provider.calls[-1][0][0].content
    assert '"name": "VISTA3D"' not in metadata


def test_nuclei_prefers_dedicated_model_and_respects_explicit_choice(client, http, pathology):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    dedicated = ModelRecord(
        project_id=project_id, name="Nuclei model", provider="threshold", label_ids=[0, 1]
    )
    with service.store.transaction() as session:
        session.insert(dedicated)
    project = service.store.get(Project, project_id)
    asset = service.store.get(Asset, asset_id)
    assert service.models.select_for_targets(project, asset, ["Nuclei"], None) == dedicated.id
    assert (
        service.models.select_for_targets(project, asset, ["Nuclei"], models["nvidia-astra"].id)
        == models["nvidia-astra"].id
    )


@pytest.mark.parametrize("project_default", [True, False])
def test_radiology_uses_vista_while_hosted_default_is_astra(http, pathology, project_default):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    project = service.store.get(Project, project_id)
    if not project_default:
        project = project.model_copy(update={"annotation_model_id": None})
    volume = service.store.get(Asset, asset_id).model_copy(update={"kind": "volume3d"})
    assert (
        service.models.select_for_targets(project, volume, ["Spleen"], None) == models["vista3d"].id
    )
    assert (
        service.models.select_for_targets(project, volume, ["Spleen"], models["nvidia-astra"].id)
        == models["nvidia-astra"].id
    )


def test_unconfigured_automatic_model_does_not_start_a_job(client, http, pathology, monkeypatch):
    project_id, asset_id, _ = pathology
    monkeypatch.delenv("TEST_ANNOTATION_KEY")
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="nuclei", name="annotate", arguments={"targets": ["Nuclei"]})],
        )
    ]
    response = http.post(
        f"/api/projects/{project_id}/assistant",
        json={"message": "Annotate nuclei", "context": {"asset_id": asset_id}},
    )
    assert response.status_code == 422
    assert "No configured model" in response.json()["detail"]
    assert client.get(f"/api/projects/{project_id}/jobs") == []


def test_automatic_does_not_escalate_to_other_presets_and_explicit_volume_choice_fails(
    http, pathology
):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    astra = models["nvidia-astra"]
    with service.store.transaction() as session:
        session.update(astra.model_copy(update={"archived": True}))
        session.update(models["custom-claude"].model_copy(update={"archived": True}))
    project = service.store.get(Project, project_id)
    asset = service.store.get(Asset, asset_id)
    with pytest.raises(DomainError, match="No configured model"):
        service.models.select_for_targets(project, asset, ["Nuclei"], None)
    with pytest.raises(DomainError, match="requires a volume"):
        service.models.select_for_targets(project, asset, ["Nuclei"], models["vista3d"].id)


def test_automatic_preserves_per_target_defaults_and_rejects_ambiguous_models(http, pathology):
    project_id, asset_id, _ = pathology
    service = http.app.state.services
    first = ModelRecord(
        project_id=project_id, name="Nuclei A", provider="threshold", label_ids=[0, 1]
    )
    second = first.model_copy(update={"id": "second-nuclei-model", "name": "Nuclei B"})
    with service.store.transaction() as session:
        session.insert(first)
        session.insert(second)
    project = service.store.get(Project, project_id)
    asset = service.store.get(Asset, asset_id)
    with pytest.raises(DomainError, match="Several models"):
        service.models.select_for_targets(project, asset, ["Nuclei"], None)
    assert (
        service.models.select_for_targets(
            project.model_copy(update={"defaults": {1: first.id}}), asset, ["Nuclei"], None
        )
        == first.id
    )
    with service.store.transaction() as session:
        session.update(second.model_copy(update={"label_ids": [0, 2]}))
    project = project.model_copy(
        update={
            "labels": project.labels + [Label(id=2, name="Cells", color="#00ff00")],
            "defaults": {1: first.id, 2: second.id},
        }
    )
    assert service.models.select_for_targets(project, asset, ["Nuclei", "Cells"], None) is None


def test_image_model_catalog_excludes_volume_models_and_other_projects(client, http, pathology):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    with service.store.transaction() as session:
        for provider, dimensions in [("medsam2", None), ("monai-unet", 3), ("monai-unet", 2)]:
            session.insert(
                ModelRecord(
                    project_id=project_id,
                    name=f"{provider}-{dimensions}",
                    provider=provider,
                    label_ids=[0, 1],
                    config={"spatial_dims": dimensions} if dimensions else {},
                )
            )
    available = client.get(f"/api/projects/{project_id}/models?asset_id={asset_id}")
    assert {m["name"] for m in available} == {
        "GPT-6 Astra",
        "Claude Opus 5",
        "My Claude",
        "monai-unet-2",
    }
    assert models["vista3d"].id in {
        m["id"] for m in client.get(f"/api/projects/{project_id}/models")
    }
    other = client.post("/api/projects", {"name": "Other"})
    assert http.get(f"/api/projects/{other['id']}/models?asset_id={asset_id}").status_code == 422


@pytest.mark.parametrize("incomplete", ["weights", "provider_model"])
def test_automatic_skips_incomplete_models_but_keeps_explicit_defaults(http, pathology, incomplete):
    project_id, asset_id, models = pathology
    service = http.app.state.services
    model = (
        ModelRecord(
            project_id=project_id,
            name="Untrained nuclei model",
            provider="monai-unet",
            label_ids=[0, 1],
            config={"spatial_dims": 2},
        )
        if incomplete == "weights"
        else ModelRecord(
            project_id=project_id,
            name="Incomplete nuclei endpoint",
            provider="openai-chat-polygons",
            label_ids=[0, 1],
            config={"url": "https://unused.test/chat/completions"},
        )
    )
    with service.store.transaction() as session:
        session.insert(model)
    project = service.store.get(Project, project_id)
    asset = service.store.get(Asset, asset_id)
    assert not service.models.configured(model)
    assert (
        service.models.select_for_targets(project, asset, ["Nuclei"], None)
        == models["nvidia-astra"].id
    )
    assert service.models.select_for_targets(project, asset, ["Nuclei"], model.id) == model.id
    assert (
        service.models.select_for_targets(
            project.model_copy(update={"annotation_model_id": model.id}), asset, ["Nuclei"], None
        )
        == model.id
    )
