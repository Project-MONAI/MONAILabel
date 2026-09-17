"""Display changes preserve annotation identities, revisions and training snapshots."""

import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.colors import default_color
from monailabel.core.models import Label, Snapshot


def test_anatomical_defaults_and_explicit_overrides():
    assert Label(id=1, name="Spleen").color == "#9d6ca2"
    assert Label(id=2, name="Liver").color == "#dd8265"
    assert Label(id=0, name="Background").color == "#000000"
    assert Label(id=1, name="Spleen", color="#00ff00").color == "#00ff00"
    assert default_color("kidney_left", 8) == default_color("Left kidney", 2)
    assert default_color("unknown finding", 1) == "#3f9c80"


def test_color_update_is_revision_checked_presentation_only(client, http, seeded):
    setup, assets = seeded
    project_id = setup["project_id"]
    prefix = f"/api/projects/{project_id}"
    project = client.get(prefix)
    asset = next(item for item in assets if item["split"] == "train")
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": client.get(f"/api/assets/{asset['id']}/fixture")["mask"],
            "covered_labels": [0, 1, 2],
        },
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    service = http.app.state.services
    snapshot = service.datasets.snapshot(project_id)
    before_assets = client.get(prefix + "/assets")
    before_models = client.get(prefix + "/models")
    mask_path = f"/api/annotations/{annotation['id']}/mask.bin"
    mask = http.get(mask_path).content
    target = project["labels"][1]
    request = {
        "targets": [target["name"].lower()],
        "color": "#AbCDeF",
        "base_version": project["version"],
    }
    updated = client.request("PATCH", prefix + "/label-colors", request)
    assert updated["labels"][1] == {**target, "color": "#abcdef"}
    assert updated == {
        **project,
        "labels": [project["labels"][0], updated["labels"][1], *project["labels"][2:]],
        "version": project["version"] + 1,
    }
    assert http.patch(prefix + "/label-colors", json=request).status_code == 409
    restored = client.request(
        "PATCH",
        prefix + "/label-colors",
        {**request, "color": None, "base_version": updated["version"]},
    )
    assert restored["labels"][1]["color"] == default_color(target["name"], target["id"])
    assert client.get(prefix + "/assets") == before_assets
    assert client.get(prefix + "/models") == before_models
    assert http.get(mask_path).content == mask
    assert service.store.get(Snapshot, snapshot.id) == snapshot


@pytest.mark.parametrize("target", ["Background", "Unknown organ", " "])
def test_color_updates_reject_nonexistent_foreground_labels(client, http, target):
    project = client.post(
        "/api/projects",
        {
            "name": "Colors",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    reply = http.patch(
        prefix + "/label-colors",
        json={"targets": [target], "color": "#0000ff", "base_version": project["version"]},
    )
    assert reply.status_code == 422
    assert client.get(prefix) == project


def test_color_prompt_persists_for_reviewer_without_inference(client, http):
    project = client.post(
        "/api/projects",
        {
            "name": "Colors",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
        },
    )
    prefix = f"/api/projects/{project['id']}"
    reviewer = client.post(
        "/api/auth/users", {"username": "color-reviewer", "password": "test-password-1234"}
    )
    client.request("PUT", prefix + "/members", {"user_id": reviewer["id"], "roles": ["reviewer"]})
    other = client.post("/api/projects", {"name": "Other"})
    http.post(
        "/api/auth/login", json={"username": "color-reviewer", "password": "test-password-1234"}
    )
    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="color",
                    name="set_label_color",
                    arguments={"targets": ["spleen"], "color": "#0000ff"},
                )
            ],
        )
    )
    reply = client.post(prefix + "/assistant", {"message": "change spleen color to blue"})
    assert reply["job_id"] is None
    assert reply["data"]["project"]["labels"][1]["color"] == "#0000ff"
    assert client.get(prefix)["labels"][1]["color"] == "#0000ff"
    assert client.get(prefix + "/jobs") == []
    http.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="restore",
                    name="set_label_color",
                    arguments={"targets": ["spleen"], "color": "anatomical"},
                )
            ],
        )
    )
    restored = client.post(prefix + "/assistant", {"message": "restore anatomical spleen color"})
    assert restored["data"]["project"]["labels"][1]["color"] == "#9d6ca2"
    assert (
        http.patch(
            f"/api/projects/{other['id']}/label-colors",
            json={"targets": ["Spleen"], "color": "#0000ff", "base_version": other["version"]},
        ).status_code
        == 403
    )
