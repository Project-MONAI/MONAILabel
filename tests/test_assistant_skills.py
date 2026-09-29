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

"""Skill loading changes model context without widening service permissions."""

from uuid import uuid4

import httpx
import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.http import HttpChat
from monailabel.server.assistant_tools import catalog
from monailabel.server.assistant_tools.base import ToolContext
from monailabel.server.instructions import parse_skill, skills


def call(tool_name, **arguments):
    return ChatMessage(
        role="assistant", tool_calls=[ToolCall(id=uuid4().hex, name=tool_name, arguments=arguments)]
    )


def test_catalog_is_valid_and_covers_only_registered_tools(http):
    from monailabel.core.models import AssistantContext, User

    registry = catalog(
        ToolContext(http.app.state.services, None, User(username="tester"), AssistantContext(), "")
    )
    names = {s.header.name for s in skills()}
    assert len(names) == len(skills())
    assert all(s.tools <= registry.tools.keys() for s in skills())
    assert set(registry.tools) == {"inspect_workspace", "clarify_request"} | set().union(
        *(s.tools for s in skills())
    )
    assert all(len(s.body.split()) < 1000 for s in skills())


@pytest.mark.parametrize(
    "source_kind,expected",
    [("volume3d", "radiology"), ("image2d", "pathology"), ("video", "video")],
)
def test_viewer_skills_match_source_geometry(source_kind, expected):
    from monailabel.server.instructions import SkillSession

    session = SkillSession(
        [], viewer=True, project=True, in_workspace=False, source_kind=source_kind
    )
    annotation_skills = {"monailabel-" + name for name in ("radiology", "pathology", "video")}
    assert session.catalog.keys() & annotation_skills == {"monailabel-" + expected}
    for name in annotation_skills - session.catalog.keys():
        with pytest.raises(DomainError, match="Choose a skill"):
            session.load(ToolCall(id="wrong-source", name="load_skill", arguments={"name": name}))
    workspace = SkillSession(
        [], viewer=True, project=True, in_workspace=True, source_kind=source_kind
    )
    assert annotation_skills <= workspace.catalog.keys()


@pytest.mark.parametrize("surface", ["workspace", "selected", "image", "video"])
def test_dataset_batch_requires_project_workspace(http, surface):
    from monailabel.core.models import AssistantContext, User, VideoPrompt
    from monailabel.server.instructions import SkillSession, coordinator_instructions

    context = AssistantContext(
        asset_id="image" if surface in {"image", "selected"} else None,
        viewer_actions=["undo"] if surface == "image" else [],
        video=(
            VideoPrompt(video_id="video", editor_id="editor", frame=0, draft_signature="0" * 64)
            if surface == "video"
            else None
        ),
    )
    registry = catalog(
        ToolContext(http.app.state.services, "project", User(username="tester"), context, "")
    )
    scope = dict(
        viewer=bool(context.asset_id),
        in_workspace=not (context.viewer_actions or context.video),
        project=True,
    )
    session = SkillSession(registry.definitions(), **scope)
    batch_skill = "monailabel-batch-annotation"
    if surface in {"workspace", "selected"}:
        assert batch_skill in session.catalog
        assert "annotate_batch" in registry.tools
    else:
        assert batch_skill not in session.catalog
        assert batch_skill not in coordinator_instructions(**scope)
        with pytest.raises(DomainError, match="not available"):
            registry.execute(
                ToolCall(id="wrong-scope", name="annotate_batch", arguments={"limit": 100})
            )
        with pytest.raises(DomainError, match="Choose a skill"):
            session.load(
                ToolCall(id="wrong-skill", name="load_skill", arguments={"name": batch_skill})
            )


@pytest.mark.parametrize("change", ["wrong-name", "../escape", "Bad-Name", "double--hyphen"])
def test_skill_rejects_invalid_or_mismatched_names(change):
    sample = next(s for s in skills() if s.header.name == "monailabel-review")
    text = "---\nname: " + change + "\ndescription: Review\n---\nInstructions\n"
    with pytest.raises(ValueError):
        parse_skill(text, sample.header.name)


def test_viewer_undo_cannot_start_a_dataset_batch(client, http, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("annotate_batch", limit=100),
        call("load_skill", name="monailabel-radiology"),
        call("viewer_edit", operation="undo"),
    ]
    reply = client.post(
        "/api/assistant",
        {
            "project_id": setup["project_id"],
            "message": "Revert the last viewer edit.",
            "context": {"asset_id": assets[0]["id"], "viewer_actions": ["undo"]},
        },
    )
    assert reply["tools"] == ["viewer_edit"]
    assert not client.get(f"/api/projects/{setup['project_id']}/jobs")


def test_native_undo_uses_the_viewer_without_calling_a_model(client, http, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    for message, operation in [("Undo that.", "undo"), ("REDO!", "redo")]:
        request = {
            "project_id": setup["project_id"],
            "request_id": uuid4().hex,
            "message": message,
            "context": {"asset_id": assets[0]["id"], "viewer_actions": ["undo", "redo"]},
        }
        reply = client.post("/api/assistant", request)
        assert reply["tools"] == ["viewer_edit"]
        assert reply["data"]["operation"] == operation
        assert client.post("/api/assistant", request) == reply
    assert not planner.calls


@pytest.mark.parametrize(
    "message", ["Undo approval", "Undo training", "Can I undo?", "Undo 2 edits"]
)
def test_native_commands_do_not_interpret_other_undo_requests(message):
    from monailabel.core.models import AssistantContext
    from monailabel.server.assistant_tools.workspace import native_viewer_command

    context = AssistantContext(asset_id="image", viewer_actions=["undo", "redo"])
    assert native_viewer_command(message, context) is None


def test_skill_loading_is_selective_and_scoped_to_the_current_turn(client, http, seeded):
    setup, _ = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="monailabel-review", purpose="answer"),
        ChatMessage(role="assistant", content="Ready"),
    ]
    body = {"project_id": setup["project_id"], "message": "Explain review decisions"}
    result = client.post("/api/assistant", body)
    initial, loaded = planner.calls[-2:]
    assert {t.name for t in initial[1]} == {"load_skill"}
    assert {t.name for t in loaded[1]} == {
        "load_skill",
        "inspect_workspace",
        "review_saved_annotations",
        "clarify_request",
    }
    assert any("<skill_instructions" in m.content for m in loaded[0] if m.role == "tool")
    assert "<name>monailabel-radiology</name>" not in initial[0][0].content
    assert not result["tools"]  # Loading guidance never claims to perform an action.
    planner.queue = [ChatMessage(role="assistant", content="Ready")]
    client.post(
        "/api/assistant", dict(body, conversation_id=result["conversation_id"], message="Hello")
    )
    assert {t.name for t in planner.calls[-1][1]} == {"load_skill"}
    assert not any("<skill_instructions" in m.content for m in planner.calls[-1][0])


def test_unknown_skill_cannot_read_local_files_or_execute_actions(client, http):
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="../../AGENTS.md"),
        ChatMessage(role="assistant", content="No"),
    ]
    result = client.post("/api/assistant", {"message": "Load a skill"})
    assert not result["tools"]
    assert {t.name for t in planner.calls[-1][1]} == {"load_skill"}
    assert "Choose a skill" in planner.calls[-1][0][-1].content


def test_listing_is_complete_after_inspection_without_requiring_a_mutation(client, http, seeded):
    setup, _ = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="monailabel-review"),
        call("inspect_workspace", collection="review_queue"),
        ChatMessage(role="assistant", content="I accepted everything."),
    ]
    result = client.post(
        "/api/assistant",
        {
            "project_id": setup["project_id"],
            "message": "Show the pending review queue",
        },
    )
    assert result["tools"] == ["inspect_workspace"]
    assert "review_queue" in result["data"]
    assert "accepted everything" not in result["message"]
    assert client.get(f"/api/projects/{setup['project_id']}/decisions") == []


def test_training_defaults_come_from_recipe_config_without_starting_work(client, http):
    from monailabel.monai.nnunet_config import NNUNetConfig
    from monailabel.monai.vista_config import VistaConfig

    planner = http.app.state.services.assistants.provider
    for recipe, expected in [
        ("vista3d", VistaConfig().model_dump(mode="json")),
        ("nnunet-v2", NNUNetConfig(modality="CT").model_dump(mode="json", exclude={"modality"})),
    ]:
        planner.queue = [
            call("load_skill", name="monailabel-workspace", purpose="answer"),
            call("inspect_workspace", collection="training_recipes", recipe_id=recipe),
            ChatMessage(role="assistant", content="Unverified or incorrect model summary"),
        ]
        result = client.post(
            "/api/assistant", {"message": f"List all the default training parameters for {recipe}"}
        )
        assert result["tools"] == ["inspect_workspace"]
        assert result["data"]["training_recipes"][0]["default_config"] == expected
        assert f"| Epochs | {expected['epochs']} |" in result["message"]
        assert f"| Steps per epoch | {expected['steps_per_epoch']} |" in result["message"]
        assert "Unverified" not in result["message"] and "learner_name" not in result["message"]
        assert result["job_id"] is None


def test_repair_cannot_substitute_a_different_mutation(client, http, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("annotate", scope="wrong"),
        call("select_annotation_model", model_id=setup["baseline_id"]),
    ]
    response = http.post(
        "/api/assistant",
        json={
            "project_id": setup["project_id"],
            "message": "Annotate this image",
            "context": {"asset_id": assets[0]["id"]},
        },
    )
    assert response.status_code == 422
    assert "No replacement action was executed" in response.json()["detail"]
    assert not client.get(f"/api/projects/{setup['project_id']}/jobs")


def test_skill_selection_has_a_bounded_planning_budget(http, client):
    planner = http.app.state.services.assistants.provider
    planner.queue = [call("load_skill", name="monailabel-workspace") for _ in range(8)]
    response = http.post("/api/assistant", json={"message": "Help"})
    assert response.status_code == 502 and len(planner.calls) == 8
    assert not client.get("/api/projects")
    contents = [m.content for m in planner.calls[-1][0] if m.role == "tool"]
    assert sum("<skill_instructions" in text for text in contents) == 1


def test_token_limit_diagnostic_is_distinct_and_omits_response_content(caplog):
    chat = HttpChat(
        CoordinatorConfig(provider="compatible", model="test", base_url="https://test/v1"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "length", "message": {"content": "private prompt text"}}
                    ],
                    "usage": {"completion_tokens": 4096, "prompt_tokens": 1200, "secret": "hidden"},
                },
            )
        ),
    )
    with caplog.at_level("INFO"), pytest.raises(DomainError, match="response token limit"):
        chat.complete([ChatMessage(role="user", content="private prompt text")], [])
    assert "4096" in caplog.text and "length" in caplog.text
    assert "private prompt" not in caplog.text and "hidden" not in caplog.text


def test_mistaking_a_skill_for_a_tool_is_repaired_without_executing_it(http, client):
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("monailabel-workspace", action="create", name="Test"),
        call("load_skill", name="monailabel-workspace"),
        call("create_project", name="Test"),
    ]
    request = {"message": "Create project Test", "request_id": uuid4().hex}
    result = client.post("/api/assistant", request)
    assert result["tools"] == ["create_project"]
    assert client.post("/api/assistant", request) == result
    assert [p["name"] for p in client.get("/api/projects")] == ["Test"]
    assert len(planner.calls) == 3


def test_named_evaluation_reference_is_project_scoped_and_unambiguous(client, http):
    from monailabel.core.evaluation import EvaluationSet
    from monailabel.core.models import AssistantContext, User
    from monailabel.server.assistant_tools.learning import evaluation_set_id

    first = client.post("/api/projects", {"name": "One"})
    second = client.post("/api/projects", {"name": "Two"})
    service = http.app.state.services
    ctx = ToolContext(service, first["id"], User(username="tester"), AssistantContext(), "")
    selected = EvaluationSet(project_id=first["id"], name="External truth")
    foreign = EvaluationSet(project_id=second["id"], name="External truth")
    with service.store.transaction() as session:
        session.insert(selected)
        session.insert(foreign)
    assert evaluation_set_id(ctx, " external TRUTH ", None) == selected.id
    with pytest.raises(DomainError, match="one exact evaluation"):
        evaluation_set_id(ctx, "External truth", foreign.id)
    duplicate = EvaluationSet(project_id=first["id"], name="External truth")
    with service.store.transaction() as session:
        session.insert(duplicate)
    with pytest.raises(DomainError, match="one exact evaluation"):
        evaluation_set_id(ctx, "External truth", None)
    assert evaluation_set_id(ctx, "External truth", selected.id) == selected.id
    with service.store.transaction() as session:
        session.update(selected.model_copy(update={"archived": True}))
    with pytest.raises(DomainError, match="one exact evaluation"):
        evaluation_set_id(ctx, "External truth", selected.id)


def test_loading_a_skill_cannot_claim_that_an_action_completed(http, client):
    from monailabel.core.chat import Conversation

    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="monailabel-workspace"),
        ChatMessage(role="assistant", content="I created the project."),
        call("create_project", name="Actual project"),
    ]
    result = client.post("/api/assistant", {"message": "Create Actual project"})
    assert result["tools"] == ["create_project"]
    assert len(client.get("/api/projects")) == 1
    saved = http.app.state.services.store.get(Conversation, result["conversation_id"])
    assert any(m.metadata.get("internal") for m in saved.turns[-1].messages)
    planner.queue = [ChatMessage(role="assistant", content="Hello")]
    client.post(
        "/api/assistant", {"message": "Hello", "conversation_id": result["conversation_id"]}
    )
    assert not any(m.metadata.get("internal") for m in planner.calls[-1][0])


def test_repeated_completion_claims_without_an_action_return_an_error(http, client):
    planner = http.app.state.services.assistants.provider
    planner.queue = [call("load_skill", name="monailabel-workspace")] + [
        ChatMessage(role="assistant", content="Done") for _ in range(3)
    ]
    response = http.post("/api/assistant", json={"message": "Create a project"})
    assert response.status_code == 502
    assert "No action was completed" in response.json()["detail"]
    assert not client.get("/api/projects")


def test_action_can_ask_for_missing_details_without_claiming_completion(http, client):
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="monailabel-workspace"),
        call("clarify_request", question="What should the project be called?"),
    ]
    result = client.post("/api/assistant", {"message": "Create a project"})
    assert result["tools"] == ["clarify_request"]
    assert "Before I can proceed:" in result["message"]
    assert not client.get("/api/projects")


def test_import_skill_loads_catalog_data_without_importing_images(client, http, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        call("load_skill", name="monailabel-dataset-import", purpose="answer"),
        ChatMessage(role="assistant", content="Here are the available datasets."),
    ]
    reply = client.post(
        "/api/assistant",
        {
            "project_id": setup["project_id"],
            "message": "Which sample datasets are available?",
        },
    )
    loaded = planner.calls[-1][0][-1]
    assert loaded.role == "tool" and "<workspace_data>" in loaded.content
    assert "Task09_Spleen" in loaded.content
    assert not reply["tools"]
    assert len(client.get(f"/api/projects/{setup['project_id']}/assets")) == len(assets)
