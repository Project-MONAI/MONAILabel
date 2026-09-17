"""Training results evaluate frozen held-out references and retain useful logs."""

import pytest
from test_evaluation_sets import accept_all, project, upload

from monailabel.core.models import Job, ModelRecord, TrainingReport


@pytest.fixture
def trained(client, http):
    pid = project(client)
    prefix = f"/api/projects/{pid}"
    for number in range(5):
        upload(http, pid, number).raise_for_status()
    client.post(prefix + "/evaluation-sets", {"name": "Fixed evaluation", "percentage": 20})
    accept_all(client, pid)
    learner = client.post(
        prefix + "/learners", {"name": "Student", "recipe": "pixel-gaussian", "label_ids": [0, 1]}
    )
    job = client.post(prefix + f"/learners/{learner['id']}/train", {})
    result = client.wait(job["id"])
    return pid, learner, job, result


def test_training_report_has_heldout_scores_logs_and_fixed_references(client, http, trained):
    pid, learner, job, result = trained
    report = client.get(f"/api/jobs/{job['id']}/training-report")
    assert report["id"] == result["training_report_id"]
    assert report["metrics"]["mean_dice"] == 1.0
    assert report["metrics"]["per_class"] == {"1": 1.0}
    assert report["per_class_iou"] == {"1": 1.0}
    assert len(report["validation_assets"]) == 1
    assert report["evaluation_version_id"]
    model = http.app.state.services.store.get(ModelRecord, result["model_id"])
    assert not set(model.training_assets) & set(report["validation_assets"])
    page = client.get(f"/api/jobs/{job['id']}/logs")
    messages = [entry["message"] for entry in page["entries"]]
    assert messages[0] == "Training queued." and messages[-1] == "Training completed. Model saved."
    assert any("held-out case 1 of 1" in message for message in messages)
    assert any("Mean Dice: 1.0000" in message for message in messages)
    tail = client.get(f"/api/jobs/{job['id']}/logs?limit=2")
    assert len(tail["entries"]) == 2 and tail["truncated"]
    exported = http.get(f"/api/jobs/{job['id']}/logs/download")
    assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
    assert "Training queued." in exported.text and "Mean Dice: 1.0000" in exported.text
    assert len(exported.text.splitlines()) == len(messages)
    assert client.get(f"/api/jobs/{job['id']}/logs?after={page['next_cursor']}")["entries"] == []
    # Frozen reference decisions stay valid after the live case returns to review.
    asset = client.get("/api/assets/" + report["validation_assets"][0])
    client.post("/api/annotations/" + asset["annotation_id"] + "/decision", {"verdict": "pending"})
    retried = client.wait(client.post(f"/api/jobs/{job['id']}/training-report")["id"])
    again = client.get(f"/api/jobs/{job['id']}/training-report")
    assert again["id"] == retried["training_report_id"] and again["metrics"] == report["metrics"]
    assert again["evaluation_version_id"] == report["evaluation_version_id"]


def test_evaluation_failure_preserves_trained_model_without_inventing_scores(
    client, http, trained, monkeypatch
):
    pid, learner, _, _ = trained

    def broken(*args, **kwargs):
        raise RuntimeError("private runtime diagnostics")

    monkeypatch.setattr(http.app.state.services.models, "predict", broken)
    job = client.post(f"/api/projects/{pid}/learners/{learner['id']}/train", {})
    result = client.wait(job["id"])
    report = client.get(f"/api/jobs/{job['id']}/training-report")
    assert report["metrics"] is None and "Evaluation failed" in report["error"]
    assert http.app.state.services.store.get(ModelRecord, result["model_id"])
    assert "private runtime diagnostics" not in str(client.get(f"/api/jobs/{job['id']}/logs"))


def test_previous_runs_can_generate_report_without_retraining_and_require_access(
    client, http, trained
):
    pid, _, job, result = trained
    store = http.app.state.services.store
    with store.transaction() as session:
        session.connection.execute(
            "DELETE FROM records WHERE kind='TrainingReport' AND project_id=?", (pid,)
        )
    assert client.get(f"/api/jobs/{job['id']}/training-report") is None
    before = len([j for j in store.list(Job, pid) if j.kind == "train"])
    evaluation_job = client.post(f"/api/jobs/{job['id']}/training-report")
    generated = client.wait(evaluation_job["id"])
    report = store.get(TrainingReport, generated["training_report_id"])
    assert report.model_id == result["model_id"] and report.metrics.mean_dice == 1.0
    assert client.get(f"/api/jobs/{evaluation_job['id']}/training-report")["id"] == report.id
    assert client.get(f"/api/jobs/{evaluation_job['id']}/logs")["entries"][0]["message"] == (
        "Evaluation queued."
    )
    assert len([j for j in store.list(Job, pid) if j.kind == "train"]) == before
    client.post("/api/auth/users", {"username": "outsider", "password": "outside-password"})
    client.post("/api/auth/login", {"username": "outsider", "password": "outside-password"})
    for suffix in ["logs", "logs/download", "training-report"]:
        assert http.get(f"/api/jobs/{job['id']}/{suffix}").status_code == 403
    assert http.post(f"/api/jobs/{job['id']}/training-report").status_code == 403


def test_compare_models_prepares_references_without_manual_snapshot(client, http, trained):
    pid, learner, _, first = trained
    second = client.wait(
        client.post(f"/api/projects/{pid}/learners/{learner['id']}/train", {})["id"]
    )
    record = client.get(f"/api/projects/{pid}/evaluation-sets")[0]
    job = client.post(
        f"/api/projects/{pid}/evaluate",
        {
            "evaluation_set_id": record["id"],
            "candidate_id": second["model_id"],
            "baseline_id": first["model_id"],
        },
    )
    result = client.wait(job["id"])
    comparison = client.get("/api/evaluations/" + result["evaluation_id"])
    assert comparison["evaluation_version_id"] == record["latest_version_id"]
    assert comparison["candidate"]["mean_dice"] == 1.0
    messages = [entry["message"] for entry in client.get(f"/api/jobs/{job['id']}/logs")["entries"]]
    assert messages[0] == "Evaluation queued."
    assert messages[-1] == "Evaluation completed. Report saved."
    assert "Evaluating model 1 of 2: Student." in messages
    assert "Evaluating model 2 of 2: Student." in messages
    assert any("held-out case 1 of 1" in message for message in messages)
    assert sum("mean Dice 1.0000" in message for message in messages) == 2
    exported = http.get(f"/api/jobs/{job['id']}/logs/download")
    assert exported.status_code == 200
    assert len(exported.text.splitlines()) == len(messages)
