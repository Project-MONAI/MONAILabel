import threading
from uuid import uuid4

import pytest

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import NotFound
from monailabel.core.models import Asset, Job, JobStatus, ModelRecord, Snapshot
from monailabel.server.deletion import cleanup_storage
from monailabel.server.jobs import Outcome


def delete_project(http, project):
    return http.request(
        "DELETE",
        f"/api/projects/{project['id']}",
        json={"confirmation_name": project["name"]},
    )


def submit(client, asset):
    mask = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
    annotation = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask, "covered_labels": [0, 1, 2]},
    )
    client.post(f"/api/annotations/{annotation['id']}/decision", {"verdict": "accepted"})
    return annotation


def test_project_delete_cascades_records_but_preserves_other_projects_and_shared_blobs(
    http, client, seeded
):
    setup, assets = seeded
    service = http.app.state.services
    project = client.get(f"/api/projects/{setup['project_id']}")
    second = client.post("/api/projects", {"name": "Keep me"})
    credential = service.secrets.save(project["id"], "Fixture credential", "not-a-real-key")
    submit(client, assets[0])
    client.post(f"/api/projects/{project['id']}/snapshots")
    planner = service.assistants.provider
    planner.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=uuid4().hex, name="inspect_workspace", arguments={"collection": "models"}
                )
            ],
        ),
        ChatMessage(role="assistant", content="Available models"),
    ]
    client.post(f"/api/projects/{project['id']}/assistant", {"message": "List models"})
    with service.store.transaction() as session:
        conversations = [
            row[0]
            for row in session.connection.execute(
                "SELECT id FROM records WHERE kind='Conversation' AND project_id=?",
                (project["id"],),
            )
        ]
        # Simulate an old receipt without project_id: it still needs cascading.
        session.connection.execute("UPDATE records SET project_id=NULL WHERE kind='ToolExecution'")
    shared = service.artifacts.put(b"shared-checkpoint")
    private = service.artifacts.put(b"private-checkpoint")
    for pid, key in [(project["id"], private), (second["id"], shared)]:
        with service.store.transaction() as session:
            session.insert(
                ModelRecord(
                    project_id=pid,
                    name="Checkpoint",
                    provider="fixture",
                    label_ids=[0],
                    state_key=service.artifacts.put_json({"weights": key}),
                )
            )
    assert delete_project(http, project).status_code == 200
    assert http.get(f"/api/projects/{project['id']}").status_code == 404
    assert http.get(f"/api/assets/{assets[0]['id']}/image").status_code == 404
    assert client.get("/api/auth/me")["username"] == "owner"
    assert client.get(f"/api/projects/{second['id']}")["name"] == "Keep me"
    with service.store.transaction() as session:
        assert not session.connection.execute(
            "SELECT id FROM records WHERE project_id=?", (project["id"],)
        ).fetchall()
        assert not session.connection.execute(
            "SELECT id FROM records WHERE kind='EncryptedCredential' AND id=?", (credential.id,)
        ).fetchall()
        assert not session.connection.execute(
            "SELECT id FROM records WHERE kind='ToolExecution' "
            "AND json_extract(data,'$.conversation_id')=?",
            (conversations[0],),
        ).fetchall()
    # Online deletion retains CAS data until a quiescent startup can collect it safely.
    assert service.artifacts.path(private).exists()
    cleanup_storage(service.store, service.artifacts)
    assert not service.artifacts.path(private).exists()
    assert service.artifacts.read(shared) == b"shared-checkpoint"


def test_asset_batch_delete_removes_selected_revisions_only(http, client, seeded):
    setup, assets = seeded
    annotation = submit(client, assets[0])
    ids = [a["id"] for a in assets[:2]]
    response = http.request(
        "DELETE", f"/api/projects/{setup['project_id']}/assets", json={"asset_ids": ids}
    )
    assert response.status_code == 200 and set(response.json()["asset_ids"]) == set(ids)
    assert len(client.get(f"/api/projects/{setup['project_id']}/assets")) == 10
    assert http.get(f"/api/annotations/{annotation['id']}/mask.bin").status_code == 404
    assert http.get(f"/api/assets/{assets[2]['id']}/image").status_code == 200
    assert client.get(f"/api/projects/{setup['project_id']}/decisions") == []


def test_snapshot_files_cannot_be_deleted_individually_and_batch_is_atomic(http, client, seeded):
    setup, assets = seeded
    submit(client, assets[0])
    snapshot = client.post(f"/api/projects/{setup['project_id']}/snapshots")
    original = http.app.state.services.store.get(Snapshot, snapshot["id"])
    response = http.request(
        "DELETE",
        f"/api/projects/{setup['project_id']}/assets",
        json={"asset_ids": [a["id"] for a in assets[:2]]},
    )
    assert response.status_code == 409 and "snapshot" in response.json()["detail"]
    assert len(client.get(f"/api/projects/{setup['project_id']}/assets")) == len(assets)
    assert http.app.state.services.store.get(Snapshot, snapshot["id"]) == original
    assert http.delete(f"/api/assets/{assets[1]['id']}").status_code == 200


@pytest.mark.parametrize("role", ["annotator", "reviewer", "outsider"])
def test_only_project_managers_can_delete(http, client, seeded, role):
    setup, assets = seeded
    project = client.get(f"/api/projects/{setup['project_id']}")
    user = client.post("/api/auth/users", {"username": role, "password": "test-password-1234"})
    if role != "outsider":
        client.request(
            "PUT",
            f"/api/projects/{project['id']}/members",
            {"user_id": user["id"], "roles": [role]},
        )
    client.post("/api/auth/login", {"username": role, "password": "test-password-1234"})
    assert delete_project(http, project).status_code == 403
    assert http.delete(f"/api/assets/{assets[0]['id']}").status_code == 403
    assert (
        http.request(
            "DELETE", f"/api/projects/{project['id']}/assets", json={"asset_ids": [assets[0]["id"]]}
        ).status_code
        == 403
    )


def test_confirmation_scope_and_active_job_guards(http, client, seeded):
    setup, assets = seeded
    project = client.get(f"/api/projects/{setup['project_id']}")
    assert delete_project(http, dict(project, name="wrong")).status_code == 422
    other = client.post("/api/demo")
    foreign = client.get(f"/api/projects/{other['project_id']}/assets")[0]
    assert (
        http.request(
            "DELETE",
            f"/api/projects/{project['id']}/assets",
            json={"asset_ids": [assets[0]["id"], foreign["id"]]},
        ).status_code
        == 422
    )
    service = http.app.state.services
    job = Job(project_id=project["id"], kind="annotate", request={}, status=JobStatus.RUNNING)
    with service.store.transaction() as session:
        session.insert(job)
    assert delete_project(http, project).status_code == 409
    assert http.delete(f"/api/assets/{assets[0]['id']}").status_code == 409
    assert len(client.get(f"/api/projects/{project['id']}/assets")) == len(assets)


def test_late_upload_snapshot_and_chat_writes_cannot_resurrect_deleted_resources(
    http, client, seeded
):
    setup, assets = seeded
    service = http.app.state.services
    asset = service.store.get(Asset, assets[0]["id"])
    assert http.delete(f"/api/assets/{asset.id}").status_code == 200
    with pytest.raises(NotFound), service.store.transaction() as session:
        session.insert(asset)
    with pytest.raises(NotFound), service.store.transaction() as session:
        session.insert(
            Job(project_id=setup["project_id"], kind="annotate", request={"asset_id": asset.id})
        )
    project = client.get(f"/api/projects/{setup['project_id']}")
    assert delete_project(http, project).status_code == 200
    with pytest.raises(NotFound), service.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=project["id"], name="Late checkpoint", provider="fixture", label_ids=[0]
            )
        )


def test_cancelled_worker_cannot_publish_after_project_deletion(http, client, seeded):
    setup, _ = seeded
    service = http.app.state.services
    project = client.get(f"/api/projects/{setup['project_id']}")
    started, finish = threading.Event(), threading.Event()
    model = ModelRecord(
        project_id=project["id"], name="Late model", provider="fixture", label_ids=[0]
    )

    def work(ctx):
        started.set()
        assert finish.wait(5)
        return Outcome({"model_id": model.id}, [model])

    try:
        job = service.jobs.submit("train", project["id"], {}, work)
        assert started.wait(5)
        service.jobs.cancel(job.id)
        assert delete_project(http, project).status_code == 200
    finally:
        finish.set()
        service.jobs.close()
    assert not service.store.list(ModelRecord, project["id"])
    assert not service.store.list(Job, project["id"])
