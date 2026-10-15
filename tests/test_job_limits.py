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

"""Reservations bound real executor work, not just nonterminal ledger entries."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Job, JobStatus, User
from monailabel.server.demo_access import DemoBusy
from monailabel.server.demo_context import acting_as
from monailabel.server.jobs import Jobs, Outcome
from monailabel.server.storage import Store


def test_queue_cancellation_cannot_bypass_capacity_and_idempotency_still_works(tmp_path):
    store = Store(tmp_path / "jobs.sqlite")
    jobs = Jobs(store, workers=1, capacity=2, per_user=2)
    started, finish = threading.Event(), threading.Event()

    def work(context):
        started.set()
        assert finish.wait(10)
        return Outcome({})

    try:
        first = jobs.submit("test", "p", {}, work, "first")
        assert started.wait(5)
        second = jobs.submit("test", "p", {}, work)
        jobs.cancel(second.id)
        for _ in range(50):
            with pytest.raises(DemoBusy) as caught:
                jobs.submit("test", "p", {}, work)
            assert caught.value.status == 429 and caught.value.retry_after == 5
        assert jobs.submit("test", "p", {}, work, "first").id == first.id
        assert len(store.list(Job)) == len(jobs.pending) == 2
    finally:
        finish.set()
        jobs.close()
    assert not jobs.pending
    assert store.get(Job, first.id).status == JobStatus.SUCCEEDED
    assert store.get(Job, second.id).status == JobStatus.CANCELLED


def test_account_quota_is_global_across_projects_and_other_accounts_keep_space(tmp_path):
    store = Store(tmp_path / "jobs.sqlite")
    users = [User(username=f"visitor-{i}") for i in range(3)]
    with store.transaction() as session:
        for user in users:
            session.insert(user)
    jobs = Jobs(store, workers=1, capacity=3, per_user=2)
    finish = threading.Event()

    def work(context):
        assert finish.wait(10)
        return Outcome({})

    try:
        with acting_as(users[0].id):
            jobs.submit("test", "first-project", {}, work)
            jobs.submit("test", "second-project", {}, work)
            with pytest.raises(DemoBusy):
                jobs.submit("test", "third-project", {}, work)
        with acting_as(users[1].id):
            jobs.submit("test", "other-user-project", {}, work)
        with acting_as(users[2].id), pytest.raises(DemoBusy):
            jobs.submit("test", "global-limit", {}, work)
        assert len(store.list(Job)) == 3
    finally:
        finish.set()
        jobs.close()


def test_parallel_submissions_reserve_atomically(tmp_path):
    store = Store(tmp_path / "jobs.sqlite")
    jobs = Jobs(store, workers=1, capacity=3, per_user=3)
    finish = threading.Event()

    def submit(index):
        try:
            return jobs.submit("test", "p", {}, lambda _: (finish.wait(10), Outcome({}))[1])
        except DemoBusy:
            return None

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(submit, range(30)))
        assert sum(result is not None for result in results) == 3
        assert len(jobs.pending) == len(store.list(Job)) == 3
    finally:
        finish.set()
        jobs.close()


def test_failed_work_releases_slot_and_closed_service_creates_no_job(tmp_path):
    store = Store(tmp_path / "jobs.sqlite")
    jobs = Jobs(store, capacity=1, per_user=1)
    released = threading.Event()
    original_release = jobs._release

    def release(identifier):
        original_release(identifier)
        released.set()

    jobs._release = release

    def work(context):
        raise DomainError("Disposable test failure.")

    try:
        first = jobs.submit("test", "p", {}, work)
        assert released.wait(5)
        assert store.get(Job, first.id).status == JobStatus.FAILED
        jobs.submit("test", "p", {}, lambda _: Outcome({}))
    finally:
        jobs.close()
    assert not jobs.pending
    with pytest.raises(DomainError, match="shutting down"):
        jobs.submit("test", "p", {}, work)
    assert len(store.list(Job)) == 2


def test_executor_failure_rolls_back_reservation_and_marks_ledger_failed(tmp_path, monkeypatch):
    store = Store(tmp_path / "jobs.sqlite")
    jobs = Jobs(store, capacity=1)

    def fail(*args, **kwargs):
        raise RuntimeError("Disposable executor failure")

    monkeypatch.setattr(jobs.executor, "submit", fail)
    try:
        with pytest.raises(DomainError, match="could not be scheduled"):
            jobs.submit("test", "p", {}, lambda _: Outcome({}))
        assert not jobs.pending
        assert store.list(Job)[0].status == JobStatus.FAILED
    finally:
        jobs.close()


def test_guard_failure_creates_no_job_or_reservation(tmp_path):
    store = Store(tmp_path / "jobs.sqlite")
    jobs = Jobs(store)

    def guard(session):
        raise DomainError("Rejected before publication")

    try:
        with pytest.raises(DomainError, match="Rejected"):
            jobs.submit("test", "p", {}, lambda _: Outcome({}), guard=guard)
        assert not jobs.pending and not store.list(Job)
    finally:
        jobs.close()
