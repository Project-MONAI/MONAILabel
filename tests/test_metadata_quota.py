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

"""Metadata quotas use disposable databases, never active demo records."""

import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from monailabel.core.chat import Conversation, ToolExecution
from monailabel.core.errors import DomainError
from monailabel.core.models import Job, JobStatus, Membership, Project, Record, Role, User
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_context import acting_as
from monailabel.server.demo_lifecycle import DemoFeedback
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.metadata_quota import JOB_RESERVE, RECORD_RESERVE, MetadataQuota
from monailabel.server.storage import Store


class Note(Record):
    project_id: str | None = None
    text: str = ""


@pytest.fixture
def database(tmp_path, monkeypatch):
    for flag in ("SELF_REGISTER", "LIFECYCLE", "STORAGE_LIMITS"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + flag, "0")
    store = Store(tmp_path / "disposable.sqlite3")
    with store.transaction() as session:
        for identifier in ("first", "second"):
            session.insert(User(id=identifier, username=identifier))
            session.insert(DemoAccess(id=identifier, minutes=60))
            MetadataQuota(session.connection).backfill(identifier)
        session.insert(User(id="operator", username="operator"))
    return store


def usage(store, owner=None):
    with store.transaction() as session:
        return MetadataQuota(session.connection).usage(owner)


def allowance(monkeypatch, store, *, owner="first", bytes_=None, records=None):
    used_bytes, used_records = usage(store, owner)
    prefix = "MONAILABEL_DEMO_METADATA_" + ("TOTAL_" if owner is None else "")
    if bytes_ is not None:
        monkeypatch.setenv(prefix + "BYTES", str(used_bytes + bytes_))
    if records is not None:
        monkeypatch.setenv(prefix + "RECORDS", str(used_records + records))


def insert(store, record, owner="first"):
    with acting_as(owner), store.transaction() as session:
        session.insert(record)


def update(store, record, owner="first"):
    with acting_as(owner), store.transaction() as session:
        session.update(record)


def full(error):
    assert error.value.status == 507
    assert error.value.code == "storage_quota_exceeded"


def test_exact_utf8_allocation_and_high_water_updates(database, monkeypatch):
    note = Note(text="ñ水🌱")
    before = usage(database, "first")
    size = len(note.model_dump_json().encode()) + RECORD_RESERVE
    allowance(monkeypatch, database, bytes_=size)
    insert(database, note)
    assert usage(database, "first") == (before[0] + size, before[1] + 1)
    grown = note.model_copy(update={"text": "x" * 300})
    update(database, grown)  # Uses already-reserved lifecycle/headroom bytes.
    with pytest.raises(DomainError) as error:
        update(database, note.model_copy(update={"text": "🌱" * 1000}))
    full(error)
    assert database.get(Note, note.id) == grown
    update(database, note)
    assert usage(database, "first") == (before[0] + size, before[1] + 1)


@pytest.mark.parametrize("owner", ["first", None])
def test_record_count_limits_and_transaction_rollback(database, monkeypatch, owner):
    allowance(monkeypatch, database, owner=owner, records=1)
    before = usage(database)
    first, second = Note(text="first"), Note(text="second")
    with acting_as("first"), pytest.raises(DomainError) as error, database.transaction() as session:
        session.insert(first)
        session.insert(second)
    full(error)
    assert database.list(Note) == []
    assert usage(database) == before
    insert(database, first)
    with pytest.raises(DomainError) as error:
        insert(database, second)
    full(error)
    assert database.list(Note) == [first]


def test_global_bytes_include_operator_data(database, monkeypatch):
    insert(database, Note(text="operator data"), "operator")
    allowance(monkeypatch, database, owner=None, bytes_=1)
    with pytest.raises(DomainError) as error:
        insert(database, Note(text="visitor data"))
    full(error)
    assert [item.text for item in database.list(Note)] == ["operator data"]


def test_deleted_records_and_restarts_do_not_refund_cumulative_allowance(database, monkeypatch):
    allowance(monkeypatch, database, records=1)
    note = Note(text="disposable")
    insert(database, note)
    consumed = usage(database, "first")
    with database.transaction() as session:
        session.connection.execute("DELETE FROM records WHERE kind='Note' AND id=?", (note.id,))
    reopened = Store(database.path)
    assert usage(reopened, "first") == consumed
    with pytest.raises(DomainError) as error:
        insert(reopened, Note(text="not an unlimited delete/recreate loop"))
    full(error)
    assert reopened.list(Note) == []


def test_claim_count_limit_covers_shared_record_writes(database, monkeypatch):
    shared = Note(text="operator prepared")
    insert(database, shared, "operator")
    with database.transaction() as session:
        count = session.connection.execute("SELECT COUNT(*) FROM metadata_claims").fetchone()[0]
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_TOTAL_CLAIMS", str(count + 1))
    update(database, shared, "first")
    with pytest.raises(DomainError) as error:
        update(database, shared, "second")
    full(error)
    update(database, shared, "first")
    assert database.get(Note, shared.id) == shared


@pytest.mark.parametrize("value", ["0", "-5", "none", str(2**63)])
def test_invalid_limits_fail_closed_before_publication(database, monkeypatch, value):
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_BYTES", value)
    with pytest.raises(DomainError) as error:
        insert(database, Note(text="denied"))
    assert error.value.status == 503
    assert database.list(Note) == []


@pytest.mark.parametrize("owner", [None, "operator"])
def test_global_limits_for_operators_are_opt_in(database, monkeypatch, owner):
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_TOTAL_BYTES", "1")
    original = Note(text="allowed operator preparation")
    insert(database, original, owner)
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "1")
    with pytest.raises(DomainError) as error:
        insert(database, Note(text="global limit enabled"), owner)
    full(error)
    update(database, original, owner)  # No new allocation, even below existing usage.


def test_parallel_transactions_cannot_overspend(database, monkeypatch):
    allowance(monkeypatch, database, records=3)

    def save(index):
        try:
            insert(database, Note(text=str(index)))
            return True
        except DomainError as error:
            assert error.status == 507
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(save, range(16))) == 3
    assert len(database.list(Note)) == 3


def test_separate_processes_cannot_overspend(database, monkeypatch):
    allowance(monkeypatch, database, records=2)
    program = """
import sys
from pathlib import Path
from monailabel.core.errors import DomainError
from monailabel.core.models import Project
from monailabel.server.demo_context import acting_as
from monailabel.server.storage import Store
store = Store(Path(sys.argv[1]))
try:
    with acting_as('first'), store.transaction() as session:
        session.insert(Project(name='disposable process test'))
except DomainError as error:
    assert error.status == 507
    sys.exit(7)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(database.path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(4)
    ]
    try:
        for child in children:
            _, errors = child.communicate(timeout=30)
            assert child.returncode in (0, 7), errors.decode()
        assert sorted(child.returncode for child in children) == [0, 0, 7, 7]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
    assert len(database.list(Project)) == 2


def test_internal_sql_updates_remain_accounted(database):
    note = Note(text="initial")
    insert(database, note)
    grown = note.model_copy(update={"text": "x" * 4096})
    before = usage(database)
    with database.transaction() as session:
        session.connection.execute(
            "UPDATE records SET data=? WHERE kind='Note' AND id=?",
            (grown.model_dump_json(), note.id),
        )
        allocated = session.connection.execute(
            "SELECT bytes FROM metadata_objects WHERE kind='Note' AND id=?", (note.id,)
        ).fetchone()[0]
    assert allocated == len(grown.model_dump_json().encode())
    assert usage(database)[0] > before[0]


def test_new_invitation_and_repeated_login_sessions_are_accounted(http, monkeypatch):
    service = http.app.state.services
    visitor = service.auth.create_user("quota-invite", "test-password-1234", demo_minutes=60)
    assert usage(service.store, visitor.id)[1] >= 4  # User, password, lease and durable visit.
    allowance(monkeypatch, service.store, owner=visitor.id, records=1)
    service.auth.issue(visitor)
    with pytest.raises(DomainError) as error:
        service.auth.issue(visitor)
    full(error)


def test_invitation_creation_is_atomic_when_initial_metadata_does_not_fit(http, monkeypatch):
    service = http.app.state.services
    before = usage(service.store)
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_RECORDS", "1")
    with pytest.raises(DomainError) as error:
        service.auth.create_user("cannot-fit", "test-password-1234", demo_minutes=60)
    full(error)
    assert not any(user.username == "cannot-fit" for user in service.store.list(User))
    assert usage(service.store) == before


def test_unauthenticated_signup_cannot_skip_starter_project_accounting(http, monkeypatch):
    for suffix in ("SELF_REGISTER", "CACHE_ONLY", "TRAINING_PREVIEW"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + suffix, "1")
    service = http.app.state.services
    before = usage(service.store)
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_RECORDS", "2")
    with pytest.raises(DomainError) as error:
        service.auth.register_demo("cannot-fit-signup", "test-password-1234")
    full(error)
    assert usage(service.store) == before
    assert not service.store.list(Project)
    assert not any(user.username == "cannot-fit-signup" for user in service.store.list(User))


def test_tool_receipt_ownership_without_actor_and_unfinished_receipt_survives(
    database, monkeypatch
):
    conversation = Conversation(user_id="first")
    insert(database, conversation)
    receipt = ToolExecution(conversation_id=conversation.id, request_hash="synthetic")
    insert(database, receipt, None)
    allowance(monkeypatch, database, records=0)
    with pytest.raises(DomainError) as error:
        insert(database, ToolExecution(conversation_id=conversation.id, request_hash="new"), None)
    full(error)
    assert database.get(ToolExecution, receipt.id).reply is None
    assert database.list(ToolExecution) == [receipt]


def test_job_publication_budget_failure_rolls_back_outcome_and_terminates(database, monkeypatch):
    # Only the job itself fits. Outcome records cannot escape the job's actor scope.
    allowance(monkeypatch, database, records=1)
    jobs = Jobs(database, workers=1)
    try:
        with acting_as("first"):
            job = jobs.submit("test", "p", {}, lambda _: Outcome({}, [Note(text="not published")]))
    finally:
        jobs.close()
    failed = database.get(Job, job.id)
    assert failed.status == JobStatus.FAILED and "storage is full" in failed.error
    assert database.list(Note) == []
    assert not jobs.pending


def test_job_result_growth_failure_preserves_terminal_status(database, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    jobs = Jobs(database, workers=1)

    def work(context):
        started.set()
        assert finish.wait(10)
        context.progress(0.5, "🌱" * 8000)
        return Outcome({"too_large": "x" * (JOB_RESERVE * 2)})

    try:
        with acting_as("first"):
            job = jobs.submit("test", "p", {}, work)
        assert started.wait(5)
        allowance(monkeypatch, database, bytes_=0)
    finally:
        finish.set()
        jobs.close()
    failed = database.get(Job, job.id)
    assert failed.status == JobStatus.FAILED and "storage is full" in failed.error
    assert failed.result == {}
    assert len(failed.progress_message) == 4000


def test_full_log_budget_does_not_block_cancellation(database, monkeypatch):
    allowance(monkeypatch, database, records=1)
    started, finish = threading.Event(), threading.Event()
    jobs = Jobs(database, workers=1)

    def work(context):
        started.set()
        assert finish.wait(10)
        return Outcome({})

    try:
        with acting_as("first"):
            job = jobs.submit("batch_annotate", "p", {}, work)
        assert started.wait(5)
        cancelled = jobs.cancel(job.id)
        assert cancelled.status == JobStatus.CANCELLED
        assert cancelled.log_count == 0
    finally:
        finish.set()
        jobs.close()
    assert database.get(Job, job.id).status == JobStatus.CANCELLED
    assert not jobs.pending


def test_restart_can_interrupt_jobs_with_no_metadata_space(database, monkeypatch):
    job = Job(project_id="p", user_id="first", kind="batch_annotate", request={})
    insert(database, job)
    allowance(monkeypatch, database, records=0, bytes_=0)
    jobs = Jobs(database)
    jobs.close()
    interrupted = database.get(Job, job.id)
    assert interrupted.status == JobStatus.INTERRUPTED
    assert interrupted.log_count == 0


def test_job_log_ownership_is_recovered_without_request_context(database, monkeypatch):
    job = Job(project_id="p", user_id="first", kind="test", request={})
    insert(database, job)
    allowance(monkeypatch, database, records=0)
    JobContext(database, job.id).log("will not fit but lifecycle update fits")
    saved = database.get(Job, job.id)
    assert saved.log_count == 0
    assert saved.progress_message == "will not fit but lifecycle update fits"


def test_legacy_schema_migration_preserves_records_and_counts_project_members(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy.sqlite3"
    owner = User(id="legacy-visitor", username="legacy-visitor")
    project = Project(name="Existing prepared project")
    rows = [
        owner,
        DemoAccess(id=owner.id, minutes=60),
        project,
        Membership(project_id=project.id, user_id=owner.id, roles=[Role.ANNOTATOR]),
        Note(project_id=project.id, text="existing note"),
    ]
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE records(kind TEXT,id TEXT,project_id TEXT,created_at TEXT,data TEXT,"
            "PRIMARY KEY(kind,id))"
        )
        for record in rows:
            connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (
                    type(record).__name__,
                    record.id,
                    getattr(record, "project_id", None),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_TOTAL_CLAIMS", "1")
    with pytest.raises(DomainError, match="increase it before startup"):
        Store(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == len(rows)
    monkeypatch.setenv("MONAILABEL_DEMO_METADATA_TOTAL_CLAIMS", "400000")
    store = Store(path)
    assert usage(store)[1] == usage(store, owner.id)[1] == len(rows)
    assert store.get(Note, rows[-1].id).text == "existing note"
    before = usage(store)
    reopened = Store(path)
    assert usage(reopened) == before  # Reopen is idempotent.


def test_metadata_failure_does_not_publish_annotation_revision(http, client, seeded, monkeypatch):
    setup, assets = seeded
    asset = assets[0]
    service = http.app.state.services
    original = client.post(
        f"/api/assets/{asset['id']}/review",
        {
            "base_revision": 0,
            "mask": np.zeros(asset["spatial_shape"], np.uint8).tolist(),
            "covered_labels": [0, 1, 2],
        },
    )
    visitor = service.auth.create_user("metadata-reviewer", "test-password-1234", demo_minutes=60)
    service.auth.add_member(setup["project_id"], visitor.id, [Role.ANNOTATOR, Role.REVIEWER])
    client.post("/api/auth/login", {"username": visitor.username, "password": "test-password-1234"})
    allowance(monkeypatch, service.store, owner=visitor.id, records=0)
    changed = np.ones(asset["spatial_shape"], np.uint8)
    response = http.post(
        f"/api/assets/{asset['id']}/review",
        json={"base_revision": 1, "mask": changed.tolist(), "covered_labels": [0, 1, 2]},
    )
    assert response.status_code == 507, response.text
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
    assert client.get(f"/api/assets/{asset['id']}/annotations") == [original]
    assert client.get(f"/api/projects/{setup['project_id']}/decisions") == []


def test_full_global_budget_rolls_back_reset_instead_of_losing_feedback(http, monkeypatch):
    service = http.app.state.services
    visitor = service.auth.create_user("reset-visitor", "test-password-1234", demo_minutes=60)
    project = Project(name="Disposable private visitor project")
    conversations = [Conversation(user_id=visitor.id, project_id=project.id) for _ in range(2)]
    with service.store.transaction() as session:
        session.insert(project)
        session.insert(Membership(project_id=project.id, user_id=visitor.id, roles=[Role.MANAGER]))
        for conversation in conversations:
            session.insert(conversation)
    service.demo_lifecycle.enabled = True
    operator = next(user for user in service.store.list(User) if user.is_admin)
    token = service.demo_lifecycle.preview()["confirmation_token"]
    before = usage(service.store)
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "1")
    allowance(monkeypatch, service.store, owner=None, records=1)
    with pytest.raises(DomainError) as error:
        service.demo_lifecycle.reset(operator, "START FRESH DEMO", token)
    full(error)
    assert service.store.get(User, visitor.id) == visitor
    assert service.store.get(Project, project.id) == project
    assert service.store.list(Conversation) == conversations
    assert service.store.list(DemoFeedback) == []
    assert usage(service.store) == before
    # Even an approved successful reset of this disposable workspace keeps its
    # allocation history, rather than manufacturing a fresh global allowance.
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "0")
    service.demo_lifecycle.reset(operator, "START FRESH DEMO", token)
    assert len(service.store.list(DemoFeedback)) == 2
    assert usage(service.store)[0] > before[0]
