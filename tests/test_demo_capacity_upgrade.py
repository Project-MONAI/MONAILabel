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

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from monailabel.core.errors import Cancelled, DomainError
from monailabel.core.models import Job, JobStatus, Project, User
from monailabel.server.auth import Auth
from monailabel.server.demo_access import DemoAccess, admission
from monailabel.server.demo_context import acting_as
from monailabel.server.demo_gpu import devices, gpu_slot, pool
from monailabel.server.jobs import Outcome


@pytest.fixture
def configured(http, monkeypatch):
    for key in ("SELF_REGISTER", "CACHE_ONLY", "TRAINING_PREVIEW", "LIFECYCLE"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + key, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_GPU_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    return http


@pytest.mark.parametrize("capacity", [4, 5])
def test_concurrent_admission_cannot_overbook(configured, monkeypatch, capacity):
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", str(capacity))
    store = configured.app.state.services.store

    def attempt(index):
        try:
            Auth(store).register_demo(f"capacity-visitor-{index}", "disposable-test-password")
            return 201
        except DomainError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=capacity + 3) as executor:
        codes = list(executor.map(attempt, range(capacity + 3)))
    assert codes.count(201) == capacity
    assert codes.count(429) == 3
    assert len(store.list(DemoAccess)) == len(store.list(Project)) == capacity
    assert admission(store)["active"] == capacity


@pytest.mark.parametrize("capacity", [4, 5])
def test_public_first_login_reserves_exactly_capacity(configured, monkeypatch, capacity):
    monkeypatch.setenv("MONAILABEL_DEMO_PUBLIC_REGISTER", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", str(capacity))
    service = configured.app.state.services
    users = []
    for index in range(capacity + 2):
        response = configured.post(
            "/api/auth/signup",
            json={"username": f"public-{index}", "password": "disposable-test-password"},
        )
        assert response.status_code == 201, response.text
        users.append(response.json())
    assert admission(service.store)["active"] == 0

    # Race the atomic admission step after authentication. HTTP sign-ins are
    # exercised separately below; their password-hash limiter intentionally
    # rejects more than four simultaneous verifications.
    def admit(user):
        try:
            service.auth.issue(service.store.get(User, user["id"]))
            return 200
        except DomainError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=capacity + 2) as executor:
        codes = list(executor.map(admit, users))
    assert codes.count(200) == capacity and codes.count(429) == 2
    assert admission(service.store)["active"] == capacity
    assert len(service.store.list(Project)) == capacity
    assert sum(lease.started_at is None for lease in service.store.list(DemoAccess)) == 2


@pytest.mark.parametrize("capacity", [4, 5])
def test_concurrent_visitors_keep_sessions_isolated_with_queued_work(
    configured, monkeypatch, capacity
):
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", str(capacity))
    monkeypatch.setenv("MONAILABEL_DEMO_PUBLIC_REGISTER", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_HTTP_LIMITS", "1")
    service = configured.app.state.services
    clients = []
    projects = []
    owners = []
    try:
        for i in range(capacity):
            client = TestClient(configured.app, client=("127.0.0.1", 51000 + i))
            response = client.post(
                "/api/auth/signup",
                json={"username": f"visitor-{i}", "password": "disposable-test-password"},
            )
            assert response.status_code == 201, response.text
            owners.append(response.json()["id"])
            assert (
                client.post(
                    "/api/auth/login",
                    json={"username": f"visitor-{i}", "password": "disposable-test-password"},
                ).status_code
                == 200
            )
            projects.append(client.get("/api/projects").json()[0]["id"])
            clients.append(client)
        assert len(set(projects)) == capacity
        for i, client in enumerate(clients):
            assert client.get(f"/api/projects/{projects[(i + 1) % capacity]}").status_code == 403
            assert (
                client.post(
                    f"/api/projects/{projects[i]}/train", json={"snapshot_id": "absent"}
                ).status_code
                == 403
            )

        lock = threading.Lock()
        running = set()
        peaks = []
        used = set()
        release = threading.Event()
        both_started = threading.Event()

        def work(context):
            with gpu_slot(context.check_cancelled) as device:
                with lock:
                    assert device not in running
                    running.add(device)
                    used.add(device)
                    peaks.append(len(running))
                    if len(running) == 2:
                        both_started.set()
                try:
                    assert release.wait(5)
                    context.check_cancelled()
                    time.sleep(0.03)
                    return Outcome({"device": device})
                finally:
                    with lock:
                        running.remove(device)

        jobs = []
        try:
            for owner, project in zip(owners, projects, strict=True):
                with acting_as(owner):
                    jobs.append(service.jobs.submit("capacity-fixture", project, {}, work))
            assert both_started.wait(5)
            assert len(service.jobs.executing()) == 2
            assert (
                sum(service.store.get(Job, j.id).status == JobStatus.QUEUED for j in jobs)
                == capacity - 2
            )

            def browse(i):
                for _ in range(10):
                    assert clients[i].get("/api/projects").status_code == 200
                    assert clients[i].get("/api/auth/session").status_code == 200

            with ThreadPoolExecutor(max_workers=capacity) as executor:
                list(executor.map(browse, range(capacity)))
        finally:
            release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(
            service.store.get(Job, j.id).status != JobStatus.SUCCEEDED for j in jobs
        ):
            time.sleep(0.02)
        assert all(service.store.get(Job, j.id).status == JobStatus.SUCCEEDED for j in jobs)
        assert max(peaks) == 2 and used == {"0", "1"}
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    finally:
        for client in clients:
            client.close()


def test_gpu_wait_is_cancellable_and_leases_survive_failure(monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_GPU_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cancel = threading.Event()
    waiting = threading.Event()

    def check():
        waiting.set()
        if cancel.is_set():
            raise Cancelled()

    def wait():
        with gpu_slot(check):
            pytest.fail("Cancelled work must not obtain a busy GPU")

    with gpu_slot(lambda: None), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait)
        assert waiting.wait(2)
        cancel.set()
        with pytest.raises(Cancelled):
            future.result(timeout=2)
    with pytest.raises(RuntimeError), gpu_slot(lambda: None):
        raise RuntimeError("disposable fixture failure")
    with gpu_slot(lambda: None) as device:
        assert device == "1"
    assert pool(("1",)).qsize() == 1


@pytest.mark.parametrize("value", ["", "0,0", "../0", "0,", "-1", "all", "3"])
def test_invalid_gpu_configuration_fails_closed(monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("MONAILABEL_DEMO_GPU_DEVICES", value)
    with pytest.raises(DomainError):
        devices()


def test_no_gpu_override_without_opt_in_and_cpu_frames_do_not_claim_gpu(monkeypatch):
    monkeypatch.delenv("MONAILABEL_DEMO_GPU_DEVICES", raising=False)
    with gpu_slot(lambda: None) as device:
        assert device is None
    monkeypatch.setenv("MONAILABEL_DEMO_GPU_DEVICES", "invalid")
    with gpu_slot(lambda: None, enabled=False) as device:
        assert device is None
