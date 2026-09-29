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

"""Shared SAM chat geometry: scope, permissions, ambiguity and fresh inference inputs."""

import gzip

import nibabel as nib
import numpy as np
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.core.models import Label, ModelRecord, Project, SliceScope, SpatialObject
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

    monkeypatch.setattr(service.models, "spatial_provider", lambda model: Predictor())
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


def test_interaction_toolbar_selects_model_without_inference(client, http, spatial_project):
    from monailabel.core.models import AssistantContext, User
    from monailabel.server.assistant_tools import catalog
    from monailabel.server.assistant_tools.base import ToolContext

    project, asset = spatial_project
    service = http.app.state.services
    context = AssistantContext(
        asset_id=asset["id"],
        base_revision=0,
        slice=SliceScope(axis=2, index=4),
        viewer_actions=["set_interaction_mode"],
        spatial_objects=[SpatialObject.model_validate(hint())],
    )
    registry = catalog(
        ToolContext(
            service,
            project["id"],
            service.store.list(User)[0],
            context,
            "Start nnInteractive for spleen.",
        )
    )
    reply = registry.execute(
        ToolCall(
            id="start",
            name="set_interaction_mode",
            arguments={
                "model_name": "nnInteractive",
                "target": "Spleen",
                "mode": "positive",
            },
        )
    )
    assert reply.job_id is None
    assert reply.data["client_action"] == "set_interaction_mode"
    assert reply.data["target"] == "Spleen"
    assert reply.data["expected"] == [hint()]
    assert service.models.get(project["id"], reply.data["model_id"]).provider == "nninteractive"
    assert service.store.get(type(registry.context.asset), asset["id"]).revision == 0
    registry.context.context = context.model_copy(
        update={"interaction_target": "Spleen", "model_id": reply.data["model_id"]}
    )
    for mode in ("negative", "box", "navigate"):
        reply = registry.execute(
            ToolCall(id=mode, name="set_interaction_mode", arguments={"mode": mode})
        )
        assert reply.data["mode"] == mode and reply.data["target"] == "Spleen"
        assert reply.job_id is None
    registry.context.context = context.model_copy(update={"viewer_actions": []})
    with pytest.raises(DomainError, match="updated .* viewer"):
        registry.execute(ToolCall(id="missing", name="set_interaction_mode", arguments={}))


def test_toolbar_update_uses_multislice_hints_without_coordinator(
    client, http, spatial_project, monkeypatch
):
    from monailabel.core.ports import Prediction

    service = http.app.state.services
    project, asset = spatial_project
    model = next(
        m for m in service.models.available(project["id"]) if m.provider == "nninteractive"
    )
    received = []

    class Predictor:
        def predict_prompted(self, image, label, model, spatial, plane, full, progress):
            received.append((spatial, full, label))
            result = np.zeros(image.shape[:-1], np.uint8)
            result[3:7, 4:8, 3:6] = label
            return Prediction(result)

    monkeypatch.setattr(service.models, "spatial_provider", lambda model: Predictor())
    body = {
        "scope": "full",
        "context": {
            "asset_id": asset["id"],
            "base_revision": 0,
            "model_id": model.id,
            "slice": {"axis": 2, "index": 4},
            "interaction_target": "Spleen",
            "spatial_objects": [
                hint(),
                hint("positive", kind="point", coordinates=[[5, 6, 5]]),
                hint("negative", kind="point", coordinates=[[10, 12, 6]], positive=False),
            ],
        },
    }
    calls_before = len(service.assistants.provider.calls)
    reply = client.post(f"/api/projects/{project['id']}/viewer-inference", body)
    client.wait(reply["job_id"])
    assert len(service.assistants.provider.calls) == calls_before
    assert received[0][1] is True
    assert [p.coordinates[2] for p in received[0][0].points] == [5, 6]
    assert [p.positive for p in received[0][0].points] == [True, False]
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
    body["context"]["base_revision"] = 9
    assert (
        http.post(f"/api/projects/{project['id']}/viewer-inference", json=body).status_code == 409
    )
    body["context"]["base_revision"] = 0
    body["context"]["spatial_objects"] = []
    assert (
        http.post(f"/api/projects/{project['id']}/viewer-inference", json=body).status_code == 422
    )


@pytest.mark.parametrize("scope", ["current_slice", "full"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_toolbar_update_without_spatial_inputs(client, http, spatial_project, scope, axis):
    service = http.app.state.services
    project, asset = spatial_project
    record = service.store.get(Project, project["id"])
    with service.store.transaction() as session:
        session.update(
            record.model_copy(
                update={"labels": [Label(id=0, name="Background"), Label(id=1, name="Spleen")]}
            )
        )
    model = ModelRecord(
        project_id=project["id"],
        name="Test threshold",
        provider="threshold",
        label_ids=[0, 1],
        config={"thresholds": [-0.5]},
    )
    with service.store.transaction() as session:
        session.insert(model)
    body = {
        "scope": scope,
        "context": {
            "asset_id": asset["id"],
            "base_revision": 0,
            "model_id": model.id,
            "slice": {"axis": axis, "index": 4},
            "spatial_prompt": {"box": [[1, 1, 4], [8, 8, 4]]},
            "spatial_objects": [hint()],
            "interaction_target": "Spleen",
        },
    }
    calls_before = len(service.assistants.provider.calls)
    reply = client.post(f"/api/projects/{project['id']}/viewer-inference", body)
    job = client.wait(reply["job_id"])
    assert len(service.assistants.provider.calls) == calls_before
    proposal = client.get(f"/api/proposals/{job['proposal_id']}")
    mask = np.frombuffer(http.get(f"/api/proposals/{proposal['id']}/mask.bin").content, np.uint8)
    mask = mask.reshape(asset["spatial_shape"])
    assert proposal["spatial_prompt"] is None
    assert np.all(np.take(mask, 4, axis=axis) == 1)
    assert np.all(np.take(mask, 0, axis=axis) == (1 if scope == "full" else 0))
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
    body["context"]["base_revision"] = 9
    assert (
        http.post(f"/api/projects/{project['id']}/viewer-inference", json=body).status_code == 409
    )


def test_toolbar_rejects_unsupported_volume_scope(client, http, spatial_project):
    project, asset = spatial_project
    model = next(
        m for m in client.get(f"/api/projects/{project['id']}/models") if m["provider"] == "sam2"
    )
    assert model["annotation_scopes"] == ["current_slice"]
    response = http.post(
        f"/api/projects/{project['id']}/viewer-inference",
        json={
            "scope": "full",
            "context": {
                "asset_id": asset["id"],
                "base_revision": 0,
                "model_id": model["id"],
                "slice": {"axis": 2, "index": 4},
                "interaction_target": "Spleen",
            },
        },
    )
    assert response.status_code == 422
    assert "does not support" in response.text


@pytest.mark.parametrize("scope", ["current_slice", "full"])
@pytest.mark.parametrize(
    "kind,positive", [("all", None), ("box", None), ("point", True), ("point", False)]
)
@pytest.mark.parametrize("target", [None, "Spleen"])
def test_clear_inputs_by_type_label_and_scope(http, spatial_project, scope, kind, positive, target):
    objects = []
    for label in ["Spleen", "Liver"]:
        for index in [4, 5]:
            objects.extend(
                [
                    hint(f"{label}-{index}-b", label, coordinates=[[2, 3, index], [12, 13, index]]),
                    hint(f"{label}-{index}-p", label, kind="point", coordinates=[[7, 8, index]]),
                    hint(
                        f"{label}-{index}-n",
                        label,
                        kind="point",
                        coordinates=[[13, 14, index]],
                        positive=False,
                    ),
                ]
            )
    args = dict(operation="clear", kind=kind, scope=scope)
    if positive is not None:
        args["positive"] = positive
    args.update({"target": target} if target else {"all_targets": True})
    response = invoke(http, spatial_project, args, objects)
    assert response.status_code == 200, response.text
    removed = set(response.json()["data"]["remove"])
    expected = {
        item["id"]
        for item in objects
        if (scope == "full" or item["coordinates"][0][2] == 4)
        and (target is None or item["target"] == target)
        and (kind == "all" or item["kind"] == kind)
        and (positive is None or item["positive"] == positive)
    }
    assert removed == expected
    assert response.json()["data"]["upsert"] == []
    asset = http.get(f"/api/assets/{spatial_project[1]['id']}").json()
    assert asset["revision"] == 0 and asset["annotation_id"] is None


def test_interaction_uses_declared_inputs_and_default_mode(
    client, http, spatial_project, monkeypatch
):
    from monailabel.core.models import AssistantContext, InteractionCapabilities, User
    from monailabel.providers import spatial
    from monailabel.server.assistant_tools import catalog
    from monailabel.server.assistant_tools.base import ToolContext

    project, asset = spatial_project
    service = http.app.state.services
    monkeypatch.setitem(
        spatial.MODELS,
        "nninteractive",
        spatial.SpatialModel(
            "nnInteractive",
            InteractionCapabilities(inputs={"box": "slice"}, output_scopes=["current_slice"]),
        ),
    )
    context = AssistantContext(
        asset_id=asset["id"],
        base_revision=0,
        slice=SliceScope(axis=2, index=4),
        viewer_actions=["set_interaction_mode"],
    )
    registry = catalog(
        ToolContext(
            service, project["id"], service.store.list(User)[0], context, "Start nnInteractive"
        )
    )
    arguments = {"model_name": "nnInteractive", "target": "Spleen"}
    reply = registry.execute(ToolCall(id="box", name="set_interaction_mode", arguments=arguments))
    assert reply.data["mode"] == "box" and reply.job_id is None
    model = client.get(f"/api/projects/{project['id']}/models")
    declared = next(m for m in model if m["id"] == reply.data["model_id"])["interaction"]
    assert declared["inputs"] == {"box": "slice"}
    assert declared["output_scopes"] == ["current_slice"]
    with pytest.raises(DomainError, match="does not support positive"):
        registry.execute(
            ToolCall(
                id="point", name="set_interaction_mode", arguments=arguments | {"mode": "positive"}
            )
        )


@pytest.mark.parametrize("coordinate", [-4, 99])
def test_outside_hints_can_be_cleared_or_moved_back_inside(http, spatial_project, coordinate):
    point = hint("outside", kind="point", coordinates=[[coordinate, 8, 4]], selected=True)
    other = hint("other", "Liver")
    response = invoke(
        http,
        spatial_project,
        {"operation": "clear", "kind": "all", "target": "Spleen", "scope": "full"},
        [point, other],
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["remove"] == ["outside"]
    moved = invoke(
        http,
        spatial_project,
        {"operation": "move", "kind": "point", "target": "Spleen", "coordinates": [[5, 8]]},
        [point, other],
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["data"]["upsert"][0]["coordinates"] == [[5, 8, 4]]
    if coordinate < 0:
        with pytest.raises(DomainError, match="Move or clear"):
            prompt_for([SpatialObject.model_validate(point)], SliceScope(axis=2, index=4), "Spleen")
