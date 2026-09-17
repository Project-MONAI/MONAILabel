import io

import pytest
from PIL import Image

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import Label

LABELS = [
    Label(id=0, name="Background", color="#000000"),
    Label(id=1, name="Nuclei", color="#00ff00"),
    Label(id=2, name="Tissue", color="#ff0000"),
]


@pytest.fixture
def image_project(client, http):
    project = client.post(
        "/api/projects",
        {"name": "Clear labels", "labels": [label.model_dump() for label in LABELS]},
    )
    out = io.BytesIO()
    Image.new("RGB", (32, 24), "pink").save(out, format="PNG")
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "clear.png", "group_id": "clear"},
        content=out.getvalue(),
    ).json()
    return project, asset


def test_clear_returns_authorized_local_action_without_inference_or_revision_change(
    client, http, image_project
):
    project, asset = image_project
    before = client.get("/api/assets/" + asset["id"])
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "remove or clear segments",
            "context": {"asset_id": asset["id"], "viewer_actions": ["clear_segments"]},
        },
    )
    assert reply["job_id"] is None
    assert reply["data"] == {
        "client_action": "clear_segments",
        "project_id": project["id"],
        "asset_id": asset["id"],
        "base_revision": 0,
        "label_ids": [1, 2],
        "image_region": None,
        "slice": None,
    }
    assert client.get("/api/assets/" + asset["id"]) == before
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.mark.parametrize("region", [None, {"x": 30, "y": 2, "width": 5, "height": 6}])
def test_missing_or_invalid_clear_region_never_falls_back_to_full_image(
    http, image_project, region
):
    project, asset = image_project
    response = http.post(
        f"/api/projects/{project['id']}/assistant",
        json={
            "message": "clear nuclei inside the selected region",
            "context": {
                "asset_id": asset["id"],
                "viewer_actions": ["clear_segments"],
                "image_region": region,
            },
        },
    )
    assert response.status_code == 422


def test_clear_region_is_retained_in_action(client, image_project):
    project, asset = image_project
    region = {"x": 4, "y": 5, "width": 6, "height": 7, "runs": [[8, 13], [14, 19]]}
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "clear nuclei inside the selected region",
            "context": {
                "asset_id": asset["id"],
                "viewer_actions": ["clear_segments"],
                "image_region": region,
            },
        },
    )
    assert reply["data"]["image_region"] == region
    assert reply["data"]["label_ids"] == [1]


def test_clear_requires_viewer_support_and_access_but_allows_reviewer_corrections(
    client, http, image_project
):
    project, asset = image_project
    body = {"message": "clear segments", "context": {"asset_id": asset["id"]}}
    reply = client.post(f"/api/projects/{project['id']}/assistant", body)
    assert "does not support" in reply["message"] and not reply["data"]
    assert "QuPath" not in reply["message"]
    other = client.post(
        "/api/projects", {"name": "Other", "labels": [label.model_dump() for label in LABELS]}
    )
    body["context"]["viewer_actions"] = ["clear_segments"]
    assert http.post(f"/api/projects/{other['id']}/assistant", json=body).status_code == 422
    reviewer = client.post(
        "/api/auth/users", {"username": "reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{project['id']}/members",
        {"user_id": reviewer["id"], "roles": ["reviewer"]},
    )
    http.post("/api/auth/login", json={"username": "reviewer", "password": "test-password-1234"})
    reply = http.post(f"/api/projects/{project['id']}/assistant", json=body)
    assert reply.status_code == 200
    assert reply.json()["data"]["label_ids"] == [1, 2]
    assert client.get("/api/assets/" + asset["id"])["revision"] == 0


def queue_clear(http, **arguments):
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="clear", name="clear_segments", arguments=arguments)],
        )
    ]


@pytest.mark.parametrize(
    "scope,all_targets", [("full", False), ("current_slice", False), ("current_slice", True)]
)
def test_volume_clear_preserves_labels_scope_and_saved_revision(
    http, client, seeded, scope, all_targets
):
    setup, assets = seeded
    project = client.get(f"/api/projects/{setup['project_id']}")
    asset = assets[0]
    labels = [label for label in project["labels"] if label["id"]]
    selected = {"axis": 1, "index": 3}
    queue_clear(
        http,
        scope=scope,
        all_targets=all_targets,
        targets=[] if all_targets else [labels[0]["name"].upper()],
    )
    before_jobs = client.get(f"/api/projects/{project['id']}/jobs")
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "Clear the requested labels",
            "context": {
                "asset_id": asset["id"],
                "base_revision": asset["revision"],
                "viewer_actions": ["clear_segments"],
                "slice": selected,
            },
        },
    )
    action = reply["data"]
    assert action["label_ids"] == [label["id"] for label in (labels if all_targets else labels[:1])]
    if scope == "current_slice":
        assert {key: action["slice"][key] for key in selected} == selected
    else:
        assert action["slice"] is None  # An existing viewer slice cannot narrow a full-volume edit.
    assert reply["job_id"] is None
    assert client.get("/api/assets/" + asset["id"]) == asset
    assert client.get(f"/api/projects/{project['id']}/jobs") == before_jobs


@pytest.mark.parametrize("selected", [None, {"axis": 2, "index": 999999}])
def test_clear_missing_or_outside_volume_slice_never_broadens_scope(http, client, seeded, selected):
    setup, assets = seeded
    queue_clear(http, scope="current_slice", all_targets=True)
    response = http.post(
        f"/api/projects/{setup['project_id']}/assistant",
        json={
            "message": "Clear all for this slice",
            "context": {
                "asset_id": assets[0]["id"],
                "viewer_actions": ["clear_segments"],
                "slice": selected,
            },
        },
    )
    assert response.status_code == 422
    assert "valid source-volume slice" in response.json()["detail"]


def test_stale_clear_and_unknown_labels_do_not_return_an_edit(http, client, seeded):
    setup, assets = seeded
    path = f"/api/projects/{setup['project_id']}/assistant"
    context = {"asset_id": assets[0]["id"], "viewer_actions": ["clear_segments"]}
    queue_clear(http, all_targets=True)
    response = http.post(
        path, json={"message": "Clear all", "context": dict(context, base_revision=999)}
    )
    assert response.status_code == 409
    for arguments in (
        {"targets": ["Unknown organ"]},
        {"targets": []},
        {"targets": ["Spleen"], "all_targets": True},
    ):
        queue_clear(http, **arguments)
        assert (
            http.post(path, json={"message": "Clear labels", "context": context}).status_code == 422
        )


def test_reviewer_can_undo_local_edits_but_cannot_submit_annotations(http, client, seeded):
    setup, assets = seeded
    project = setup["project_id"]
    reviewer = client.post(
        "/api/auth/users", {"username": "reviewer", "password": "test-password-1234"}
    )
    client.request(
        "PUT",
        f"/api/projects/{project}/members",
        {"user_id": reviewer["id"], "roles": ["reviewer"]},
    )
    client.post("/api/auth/login", {"username": "reviewer", "password": "test-password-1234"})
    for operation, status in (("undo", 200), ("redo", 200), ("submit", 403)):
        http.app.state.services.assistants.provider.queue = [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id=operation, name="viewer_edit", arguments={"operation": operation})
                ],
            )
        ]
        response = http.post(
            f"/api/projects/{project}/assistant",
            json={
                "message": operation,
                "context": {"asset_id": assets[0]["id"], "viewer_actions": [operation]},
            },
        )
        assert response.status_code == status
