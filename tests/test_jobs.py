import threading
import time

import pytest

from monailabel.core.errors import Conflict
from monailabel.core.models import Job, JobStatus, ModelRecord
from monailabel.server.jobs import Jobs, Outcome
from monailabel.server.storage import Store


def wait(store, identifier):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = store.get(Job, identifier)
        if job.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    pytest.fail("Job did not finish")


@pytest.mark.parametrize("kind", ["train", "evaluate"])
def test_cancellation_publishes_no_records_and_idempotency(tmp_path, kind):
    store = Store(tmp_path / "db.sqlite")
    jobs = Jobs(store, workers=1)
    started, finish = threading.Event(), threading.Event()
    model = ModelRecord(
        project_id="project", name="Never publish", provider="test", label_ids=[0, 1]
    )

    def work(context):
        started.set()
        assert finish.wait(5)
        return Outcome({"model_id": model.id}, [model])

    try:
        job = jobs.submit(kind, "project", {"round": 1}, work, "same-key")
        assert started.wait(5)
        assert jobs.submit(kind, "project", {"round": 1}, work, "same-key").id == job.id
        with pytest.raises(Conflict):
            jobs.submit(kind, "project", {"round": 2}, work, "same-key")
        jobs.cancel(job.id)
        finish.set()
    finally:
        finish.set()
        jobs.close()
    assert store.get(Job, job.id).status == JobStatus.CANCELLED
    assert not store.list(ModelRecord)
    title = "Training" if kind == "train" else "Evaluation"
    assert jobs.logs(job.id).entries[-1].message == f"{title} cancelled."


@pytest.mark.parametrize("kind", ["train", "evaluate"])
def test_restart_marks_interrupted_and_keeps_finished_jobs(tmp_path, kind):
    store = Store(tmp_path / "db.sqlite")
    running = Job(project_id="p", kind=kind, request={}, status=JobStatus.RUNNING)
    complete = Job(project_id="p", kind=kind, request={}, status=JobStatus.SUCCEEDED)
    with store.transaction() as session:
        session.insert(running)
        session.insert(complete)
    jobs = Jobs(store)
    jobs.close()
    assert store.get(Job, running.id).status == JobStatus.INTERRUPTED
    assert store.get(Job, complete.id).status == JobStatus.SUCCEEDED
    title = "Training" if kind == "train" else "Evaluation"
    assert jobs.logs(running.id).entries[-1].message == f"{title} interrupted by server restart."


@pytest.mark.parametrize("kind", ["train", "evaluate", "training_report"])
def test_success_publishes_result_and_record_together(tmp_path, kind):
    store = Store(tmp_path / "db.sqlite")
    jobs = Jobs(store)
    model = ModelRecord(project_id="p", name="Published", provider="test", label_ids=[0, 1])
    try:
        job = jobs.submit(kind, "p", {}, lambda context: Outcome({"model_id": model.id}, [model]))
        assert wait(store, job.id).result == {"model_id": model.id}
        assert store.get(ModelRecord, model.id) == model
        title, saved = ("Training", "Model") if kind == "train" else ("Evaluation", "Report")
        assert [entry.message for entry in jobs.logs(job.id).entries] == [
            f"{title} queued.",
            f"{title} started.",
            f"{title} completed. {saved} saved.",
        ]
    finally:
        jobs.close()


def test_evaluation_failure_has_readable_error_log(tmp_path):
    from monailabel.core.errors import DomainError

    store = Store(tmp_path / "db.sqlite")
    jobs = Jobs(store)

    def work(context):
        context.log("Evaluating held-out case 1 of 2.")
        raise DomainError("Evaluation reference is unavailable.")

    try:
        job = jobs.submit("evaluate", "p", {}, work)
        assert wait(store, job.id).status == JobStatus.FAILED
        entry = jobs.logs(job.id).entries[-1]
        assert entry.level == "error" and entry.message == "Evaluation reference is unavailable."
    finally:
        jobs.close()


def test_log_tail_full_download_cursor_and_cancellation(tmp_path, monkeypatch):
    import monailabel.server.jobs as module
    from monailabel.core.errors import Cancelled
    from monailabel.server.jobs import JobContext

    monkeypatch.setattr(module, "LOG_LIMIT", 3)
    store = Store(tmp_path / "db.sqlite")
    jobs = Jobs(store)
    job = Job(project_id="p", kind="train", request={}, status=JobStatus.RUNNING)
    with store.transaction() as session:
        session.insert(job)
    context = JobContext(store, job.id)
    try:
        for index in range(5):
            context.log(f"Step {index}")
        context.progress(0.5)
        assert store.get(Job, job.id).progress_message == "Step 4"
        page = jobs.logs(job.id)
        assert page.truncated and [e.sequence for e in page.entries] == [3, 4, 5]
        assert page.total_lines == 5
        text = "".join(jobs.download_log(job.id))
        assert len(text.splitlines()) == 5 and "Step 0" in text and "Step 4" in text
        assert [e.sequence for e in jobs.logs(job.id, 4).entries] == [5]
        jobs.cancel(job.id)
        with pytest.raises(Cancelled):
            context.log("Must not append after cancellation")
        assert jobs.logs(job.id, page.next_cursor).entries[0].message == "Training cancelled."
    finally:
        jobs.close()
