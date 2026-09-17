"""Named training resolves within a project and cannot fall back to stale chat selection."""

import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import Job, Learner


def queue(http, args):
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="train", name="start_training", arguments=args)],
        )
    ]


@pytest.fixture
def named_setups(client, http, seeded):
    project_id = seeded[0]["project_id"]
    service = http.app.state.services
    names = ["Spleen specialist", "Liver specialist"]
    learners = [
        Learner(
            project_id=project_id,
            protocol_version=1,
            name=name,
            recipe="pixel-gaussian",
            label_ids=[0, 1, 2],
        )
        for name in names
    ]
    with service.store.transaction() as session:
        for learner in learners:
            session.insert(learner)
    return project_id, learners


def test_named_training_overrides_context_and_sets_followup_selection(
    client, http, named_setups, monkeypatch
):
    project_id, learners = named_setups
    calls = []

    def start(project_id, learner_id, request, *, authorized_by):
        assert authorized_by
        calls.append((project_id, learner_id, request))
        return Job(kind="train", project_id=project_id, request={})

    monkeypatch.setattr(http.app.state.services.learning, "start", start)
    queue(http, {"learner_name": "  SPLEEN SPECIALIST "})
    reply = client.post(
        f"/api/projects/{project_id}/assistant",
        {"message": "Train Spleen specialist", "context": {"learner_id": learners[1].id}},
    )
    assert calls[0][1] == learners[0].id
    assert reply["data"]["learner_id"] == learners[0].id
    assert "Spleen specialist" in reply["message"]


@pytest.mark.parametrize("ambiguity", ["unknown", "duplicate", "mismatched_id", "other_project"])
def test_unclear_names_ask_without_starting_any_job(
    client, http, named_setups, monkeypatch, ambiguity
):
    project_id, learners = named_setups
    service = http.app.state.services
    args = {"learner_name": "Spleen specialist"}
    if ambiguity == "unknown":
        args["learner_name"] = "No such model"
    elif ambiguity == "duplicate":
        with service.store.transaction() as session:
            session.insert(
                Learner(
                    project_id=project_id,
                    protocol_version=1,
                    name="spleen specialist",
                    recipe="pixel-gaussian",
                    label_ids=[0, 1, 2],
                )
            )
    elif ambiguity == "mismatched_id":
        args["learner_id"] = learners[1].id
    else:
        project_id = client.post("/api/projects", {"name": "Different project"})["id"]

    def never(*args, **kwargs):
        raise AssertionError("Ambiguous named request must not start training")

    monkeypatch.setattr(service.learning, "start", never)
    queue(http, args)
    reply = client.post(
        f"/api/projects/{project_id}/assistant",
        {"message": "Train Spleen specialist", "context": {"learner_id": learners[1].id}},
    )
    assert reply["job_id"] is None and "Which" in reply["message"]
    assert client.get(f"/api/projects/{project_id}/jobs") == []


def test_user_name_is_retained_on_created_setup_without_starting_training(client, http):
    http.app.state.services.presets.enabled = True
    project = client.post("/api/projects", {"name": "Custom named models"})
    http.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="create",
                    name="create_learner",
                    arguments={
                        "recipe": "vista3d",
                        "initialization": "fine_tune",
                        "name": 'Spleen "pilot"',
                        "targets": ["spleen"],
                    },
                )
            ],
        )
    ]
    reply = client.post(
        f"/api/projects/{project['id']}/assistant",
        {"message": 'Create a VISTA3D model named Spleen "pilot"; do not train'},
    )
    learner = client.get(f"/api/projects/{project['id']}/learners")[0]
    assert learner["name"] == 'Spleen "pilot"'
    assert reply["data"]["learner_id"] == learner["id"] and reply["job_id"] is None
    assert client.get(f"/api/projects/{project['id']}/jobs") == []
