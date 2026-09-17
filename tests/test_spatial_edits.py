"""Shared SAM chat geometry: scope, permissions, ambiguity and fresh inference inputs."""

import gzip

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import SliceScope, SpatialObject
from monailabel.server.assistant_tools.spatial import prompt_for


@pytest.fixture
def spatial_project(client, http):
    http.app.state.services.presets.enabled = True
    project = client.post("/api/projects", {"name": "Spatial controls"})
    content = gzip.compress(nib.Nifti1Image(np.zeros((20, 22, 10), np.int16), np.eye(4)).to_bytes())
    asset = http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "source.nii.gz"},
        content=content,
    ).json()
    return project, asset


def hint(identifier="b", target="Spleen", kind="box", coordinates=None, **kwargs):
    return SpatialObject(
        id=identifier,
        target=target,
        kind=kind,
        coordinates=coordinates or [[2, 3, 4], [12, 13, 4]],
        **kwargs,
    ).model_dump(mode="json")


def invoke(http, fixture, args, objects=(), **context):
    project, asset = fixture
    args = dict(args)
    positive = args.pop("positive", None)
    args["polarity"] = (
        ("positive" if positive else "negative")
        if positive is not None
        else "positive"
        if args["operation"] == "add" and args.get("kind", "point") == "point"
        else "all"
    )
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="spatial", name="edit_spatial_prompts", arguments=args)],
        )
    ]
    return http.post(
        f"/api/projects/{project['id']}/assistant",
        json={
            "message": "Edit my SAM hints " + str(args.get("coordinates", "")),
            "context": {
                "asset_id": asset["id"],
                "base_revision": asset["revision"],
                "slice": {"axis": 2, "index": 4, "window": [0, 100]},
                "viewer_actions": ["edit_spatial_prompts"],
                "spatial_objects": list(objects),
                **context,
            },
        },
    )


def test_box_point_move_and_clear_are_local_and_preserve_other_targets(
    client, http, spatial_project
):
    project, asset = spatial_project
    box = hint()
    other = hint("other", "Liver")
    response = invoke(
        http,
        spatial_project,
        {"operation": "add", "target": "Spleen", "box_center": True, "positive": True},
        [box, other],
    )
    assert response.status_code == 200, response.text
    action = response.json()["data"]
    point = action["upsert"][0]
    assert point["coordinates"] == [[7, 8, 4]] and point["positive"]
    assert action["expected"] == [box, other] and action["remove"] == []
    assert "starting point" in response.json()["message"]
    negative = hint("neg", kind="point", coordinates=[[14, 15, 4]], positive=False)
    away = hint("away", kind="point", coordinates=[[14, 15, 5]], positive=False)
    objects = [box, other, point, negative, away]
    moved = invoke(
        http,
        spatial_project,
        {"operation": "move", "target": "Spleen", "positive": True, "coordinates": [[8, 9]]},
        objects,
    ).json()["data"]["upsert"][0]
    assert moved["id"] == point["id"] and moved["coordinates"] == [[8, 9, 4]]
    response = invoke(
        http,
        spatial_project,
        {"operation": "clear", "kind": "point", "positive": False, "all_targets": True},
        objects,
    )
    assert response.json()["data"]["remove"] == ["neg"]
    response = invoke(
        http, spatial_project, {"operation": "clear", "kind": "all", "target": "Spleen"}, objects
    )
    assert set(response.json()["data"]["remove"]) == {"b", point["id"], "neg"}
    response = invoke(
        http,
        spatial_project,
        {"operation": "clear", "kind": "all", "target": "Spleen", "scope": "full"},
        objects,
    )
    assert (
        "away" in response.json()["data"]["remove"]
        and "other" not in response.json()["data"]["remove"]
    )
    assert client.get("/api/assets/" + asset["id"]) == asset
    assert client.get(f"/api/projects/{project['id']}/jobs") == []


@pytest.mark.parametrize(
    "args,context",
    [
        ({"operation": "add"}, {}),
        ({"operation": "add", "coordinates": [[20, 3]]}, {}),
        ({"operation": "add", "coordinates": [[2, 3, 5]]}, {}),
        ({"operation": "add", "kind": "box", "coordinates": [[8, 9], [2, 3]]}, {}),
        ({"operation": "add", "box_center": True}, {}),
        ({"operation": "move", "kind": "box", "coordinates": [[1, 2], [5, 6]]}, {}),
        ({"operation": "clear", "kind": "all"}, {}),
        ({"operation": "clear", "all_targets": True}, {"viewer_actions": []}),
        ({"operation": "clear", "all_targets": True}, {"slice": None}),
    ],
)
def test_missing_or_invalid_geometry_never_guesses(http, spatial_project, args, context):
    response = invoke(http, spatial_project, args, **context)
    assert response.status_code == 422, response.text


def test_stale_revision_and_foreign_asset_refuse_edits(client, http, spatial_project):
    response = invoke(
        http, spatial_project, {"operation": "clear", "all_targets": True}, base_revision=99
    )
    assert response.status_code == 409
    other = client.post("/api/projects", {"name": "Other"})
    response = invoke(
        http, (other, spatial_project[1]), {"operation": "clear", "all_targets": True}
    )
    assert response.status_code == 422


def test_inference_combines_target_box_and_both_point_polarities():
    items = [
        SpatialObject.model_validate(i)
        for i in [
            hint(),
            hint("p", kind="point", coordinates=[[7, 8, 4]]),
            hint("n", kind="point", coordinates=[[13, 14, 4]], positive=False),
            hint("other", "Liver"),
            hint("off", kind="point", coordinates=[[7, 8, 5]]),
        ]
    ]
    scope = SliceScope(axis=2, index=4, window=[0, 100])
    prompt = prompt_for(items, scope, "Spleen")
    assert prompt.box == [[2, 3, 4], [12, 13, 4]]
    assert [(p.coordinates, p.positive) for p in prompt.points] == [
        ([7, 8, 4], True),
        ([13, 14, 4], False),
    ]
    items.append(SpatialObject.model_validate(hint("duplicate")))
    with pytest.raises(DomainError, match="Select exactly one"):
        prompt_for(items, scope, "Spleen")
    items[-1] = items[-1].model_copy(
        update={"selected": True, "coordinates": [[1, 1, 4], [6, 6, 4]]}
    )
    assert prompt_for(items, scope, "Spleen").box == [[1, 1, 4], [6, 6, 4]]


def test_coordinates_must_come_from_current_request(http, spatial_project):
    from monailabel.core.models import AssistantContext, User
    from monailabel.server.assistant_tools.base import ToolContext
    from monailabel.server.assistant_tools.spatial import SpatialArgs, edit

    service = http.app.state.services
    project, asset = spatial_project
    context = ToolContext(
        service,
        project["id"],
        service.store.list(User)[0],
        AssistantContext(
            asset_id=asset["id"],
            base_revision=0,
            slice=SliceScope(axis=2, index=4, window=[0, 100]),
            viewer_actions=["edit_spatial_prompts"],
            spatial_objects=[SpatialObject.model_validate(hint())],
        ),
        "Add a point somewhere in spleen",
    )
    with pytest.raises(DomainError, match="copied from the user's current request"):
        edit(context, SpatialArgs(operation="add", polarity="positive", coordinates=[7, 8]))
    context.message = "Add a point at voxel 7, 8"
    assert edit(
        context, SpatialArgs(operation="add", polarity="positive", coordinates=[7, 8])
    ).data["upsert"][0]["coordinates"] == [[7, 8, 4]]


@pytest.mark.parametrize(
    "roles,allowed",
    [(["annotator"], True), (["reviewer"], True), (["manager"], True), ([], False)],
)
def test_spatial_edits_require_edit_permission(client, http, spatial_project, roles, allowed):
    project, _ = spatial_project
    user = client.post(
        "/api/auth/users", {"username": "hint-user", "password": "test-password-1234"}
    )
    if roles:
        client.request(
            "PUT", f"/api/projects/{project['id']}/members", {"user_id": user["id"], "roles": roles}
        )
    http.post("/api/auth/login", json={"username": "hint-user", "password": "test-password-1234"})
    response = invoke(
        http, spatial_project, {"operation": "clear", "kind": "all", "all_targets": True}
    )
    assert response.status_code == (200 if allowed else 403), response.text


def test_named_sam_overrides_default_and_receives_combined_fresh_geometry(
    client, http, spatial_project, monkeypatch
):
    from monailabel.core.ports import Prediction

    service = http.app.state.services
    project, asset = spatial_project
    calls = []

    class Predictor:
        def predict_prompted(self, image, label, model, spatial, plane, full, progress):
            calls.append((model.provider, spatial))
            return Prediction(np.zeros(image.shape[:-1], np.uint8))

    monkeypatch.setattr(service.models, "spatial_provider", lambda: Predictor())
    service.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="infer",
                    name="annotate",
                    arguments={
                        "model_name": "SAM 2.1",
                        "targets": ["Spleen"],
                        "scope": "current_slice",
                    },
                )
            ],
        )
    ]
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {
            "message": "Segment spleen using SAM 2.1",
            "context": {
                "asset_id": asset["id"],
                "base_revision": 0,
                "model_id": project["annotation_model_id"],
                "slice": {"axis": 2, "index": 4, "window": [0, 100]},
                "spatial_objects": [
                    hint(),
                    hint("p", kind="point", coordinates=[[7, 8, 4]]),
                    hint("n", kind="point", coordinates=[[15, 16, 4]], positive=False),
                ],
            },
        },
    )
    result = client.wait(reply["job_id"])
    proposal = client.get("/api/proposals/" + result["proposal_id"])
    assert len(calls) == 1 and calls[0][0] == "sam2"
    assert proposal["spatial_prompt"]["box"] == [[2, 3, 4], [12, 13, 4]]
    assert {p["positive"] for p in proposal["spatial_prompt"]["points"]} == {True, False}
    assert client.get("/api/assets/" + asset["id"]) == asset
