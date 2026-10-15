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

"""Bounded spools and crash scenarios use disposable test directories only."""

import io
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, Job, JobStatus, User
from monailabel.server.demo_access import DemoAccess
from monailabel.server.jobs import JobContext
from monailabel.server.storage import Store
from monailabel.server.temporary_storage import TemporaryReservations, job_spool


@pytest.fixture
def reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "monailabel.server.temporary_storage.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1024**4),
    )
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "0")
    return TemporaryReservations(tmp_path / "temporary")


def held(reservations):
    with reservations.connection() as connection:
        return connection.execute("SELECT id,owner,bytes FROM reservations").fetchall()


def test_aggregate_extent_limit_counts_rewrites_and_sparse_seeks(reservations):
    with reservations.reserve("visitor", 16) as spool:
        with spool.open("zip") as first:
            assert first.write(b"a" * 8) == 8
            first.seek(0)
            first.write(b"b" * 8)  # ZIP headers can be rewritten without double charging.
            with pytest.raises(io.UnsupportedOperation):
                first.fileno()
            with spool.open("mask") as second:
                second.write(b"m" * 8)
                first.seek(100)
                with pytest.raises(DomainError) as error:
                    first.write(b"!")
                assert error.value.code == "storage_quota_exceeded"
                with pytest.raises(ValueError, match="open"):
                    spool.remove("mask")
            assert sum(p.stat().st_size for p in spool.directory.iterdir()) == 16
            spool.remove("mask")
            first.seek(8)
            first.write(b"c" * 8)
        assert spool.path("zip").read_bytes() == b"b" * 8 + b"c" * 8
    assert held(reservations) == []
    assert not spool.directory.exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", ".", "..", "nested/file", "a\\b"])
def test_spool_refuses_nonprivate_names(reservations, name):
    with reservations.reserve("visitor", 16) as spool, pytest.raises(ValueError), spool.open(name):
        pass


@pytest.mark.parametrize("encoder", ["numpy", "pillow", "zip"])
def test_native_encoders_cannot_bypass_checked_writes(reservations, encoder):
    with reservations.reserve("visitor", 512) as spool:
        with pytest.raises(DomainError) as error, spool.open("output") as stream:
            array = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
            if encoder == "numpy":
                np.save(stream, array, allow_pickle=False)
            elif encoder == "pillow":
                Image.fromarray(array).save(stream, format="PNG")
            else:
                with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                    archive.writestr("array", array.tobytes())
        assert error.value.status == 507
        assert spool.path("output").stat().st_size <= 512
    assert held(reservations) == []


@pytest.mark.parametrize("encoder", ["numpy", "pillow"])
def test_successful_native_encoding_preserves_values(reservations, encoder):
    array = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    with reservations.reserve("visitor", 8192) as spool:
        with spool.open("output") as stream:
            if encoder == "numpy":
                np.save(stream, array, allow_pickle=False)
            else:
                Image.fromarray(array).save(stream, format="PNG")
        if encoder == "numpy":
            decoded = np.load(spool.path("output"), allow_pickle=False)
        else:
            with Image.open(spool.path("output")) as image:
                decoded = np.asarray(image)
        assert np.array_equal(array, decoded)
        assert spool.sizes["output"] == spool.path("output").stat().st_size


def test_zip_roundtrip_and_successful_file_reuse(reservations):
    with reservations.reserve("visitor", 10000) as spool:
        for _ in range(2):
            with spool.open("data.zip") as stream, ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                archive.writestr("a", b"a" * 10000)
                archive.writestr("b", b"b" * 10000)
            with ZipFile(spool.path("data.zip")) as archive:
                assert archive.testzip() is None
                assert archive.read("a") == b"a" * 10000
            assert spool.sizes["data.zip"] == spool.path("data.zip").stat().st_size
            spool.remove("data.zip")
        assert spool.sizes == {}


@pytest.mark.parametrize(
    "setting,first,second",
    [
        ("TEMP_TOTAL_BYTES", "first", "second"),
        ("TEMP_VISITOR_BYTES", "first", "first"),
        ("TEMP_TOTAL_SPOOLS", "first", "second"),
        ("TEMP_VISITOR_SPOOLS", "first", "first"),
    ],
)
def test_reservations_limit_bytes_and_slots_across_instances(
    reservations, monkeypatch, setting, first, second
):
    monkeypatch.setenv("MONAILABEL_DEMO_" + setting, "16" if setting.endswith("BYTES") else "1")
    with reservations.reserve(first, 16):
        reopened = TemporaryReservations(reservations.root)
        with pytest.raises(DomainError), reopened.reserve(second, 1):
            pass
        assert len(held(reservations)) == 1
    with reservations.reserve(second, 1):
        assert len(held(reservations)) == 1


def test_cleanup_failure_retains_reservation_and_only_own_directory(reservations, monkeypatch):
    original = __import__("shutil").rmtree
    paths = []

    def fail(path):
        paths.append(path)
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr("monailabel.server.temporary_storage.shutil.rmtree", fail)
    with (
        pytest.raises(OSError, match="cleanup"),
        reservations.reserve("visitor", 16) as spool,
        spool.open("input") as stream,
    ):
        stream.write(b"retained")
    assert paths == [spool.directory]
    assert held(reservations) == [(spool.directory.name, "visitor", 16)]
    assert spool.path("input").read_bytes() == b"retained"
    monkeypatch.setattr("monailabel.server.temporary_storage.shutil.rmtree", original)


def test_disk_headroom_rejection_does_not_reserve(reservations, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "20")
    monkeypatch.setattr(
        "monailabel.server.temporary_storage.shutil.disk_usage",
        lambda path: SimpleNamespace(free=30),
    )
    with pytest.raises(DomainError), reservations.reserve("visitor", 11):
        pass
    assert held(reservations) == []


def test_process_crash_cannot_refund_or_reuse_occupied_capacity(reservations, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_TEMP_TOTAL_BYTES", "16")
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "1")
    script = """
import os, sys
from pathlib import Path
from monailabel.server.temporary_storage import TemporaryReservations
with TemporaryReservations(Path(sys.argv[1])).reserve('visitor',16) as spool:
    with spool.open('input') as stream:
        stream.write(b'crashed')
    os._exit(7)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(reservations.root)], timeout=30, capture_output=True
    )
    assert result.returncode == 7, result.stderr.decode()
    reopened = TemporaryReservations(reservations.root)
    rows = held(reopened)
    assert len(rows) == 1 and rows[0][1:] == ("visitor", 16)
    assert (reservations.root / rows[0][0] / "input").read_bytes() == b"crashed"
    with pytest.raises(DomainError), reopened.reserve("other", 1):
        pass


def test_parallel_reservations_admit_only_one_writer(reservations, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_TEMP_TOTAL_BYTES", "16")
    barrier = threading.Barrier(4, timeout=10)

    def attempt(index):
        barrier.wait()
        try:
            with TemporaryReservations(reservations.root).reserve(str(index), 16):
                barrier.wait()
                return True
        except DomainError:
            barrier.wait()
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(attempt, range(4))) == 1
    assert held(reservations) == []


def test_separate_process_respects_committed_reservation(reservations, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_TEMP_TOTAL_BYTES", "16")
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "1")
    script = """
import sys
from pathlib import Path
from monailabel.core.errors import DomainError
from monailabel.server.temporary_storage import TemporaryReservations
try:
    with TemporaryReservations(Path(sys.argv[1])).reserve('other',1):
        sys.exit(9)
except DomainError as error:
    assert error.status == 507
"""
    with reservations.reserve("visitor", 16):
        result = subprocess.run(
            [sys.executable, "-c", script, str(reservations.root)], timeout=30, capture_output=True
        )
        assert result.returncode == 0, result.stderr.decode()


def test_job_spool_uses_persisted_lease_without_actor_or_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_LIFECYCLE", "0")
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "0")
    monkeypatch.setenv("MONAILABEL_DEMO_EXPORT_TEMP_BYTES", "8")
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "1")
    store = Store(tmp_path / "workspace.sqlite3")
    with store.transaction() as session:
        session.insert(User(id="visitor", username="visitor"))
        session.insert(DemoAccess(id="visitor", minutes=60))
        session.insert(
            Job(
                id="visitor-job",
                user_id="visitor",
                kind="dataset_export",
                project_id="p",
                request={},
            )
        )
        session.insert(Job(id="operator-job", kind="dataset_export", project_id="p", request={}))
    with job_spool(JobContext(store, "visitor-job"), "EXPORT_TEMP_BYTES", 100) as spool:
        assert spool.capacity == 8
    with job_spool(JobContext(store, "operator-job"), "EXPORT_TEMP_BYTES", 100) as spool:
        assert spool.capacity is None
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "1")
    with job_spool(JobContext(store, "operator-job"), "EXPORT_TEMP_BYTES", 100) as spool:
        assert spool.capacity == 8
    monkeypatch.setenv("MONAILABEL_DEMO_EXPORT_TEMP_BYTES", "invalid")
    with (
        pytest.raises(DomainError) as error,
        job_spool(JobContext(store, "visitor-job"), "EXPORT_TEMP_BYTES", 100),
    ):
        pass
    assert error.value.status == 503


def wait_job(store, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = store.get(Job, identifier)
        if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return job
        time.sleep(0.01)
    pytest.fail("Disposable export did not finish")


@pytest.mark.parametrize("derived", [False, True])
def test_export_quota_failure_preserves_revision_and_successful_retry(
    http, client, seeded, monkeypatch, derived
):
    setup, assets = seeded
    asset = assets[0]
    service = http.app.state.services
    saved = service.store.get(Asset, asset["id"])
    if derived:
        with service.store.transaction() as session:
            session.update(saved.model_copy(update={"source_key": None}))
    mask = np.zeros(service.artifacts.array(saved.image_key).shape[:-1], np.uint8)
    response = http.post(
        f"/api/assets/{asset['id']}/review-mask",
        params={"base_revision": 0, "covered_labels": [1]},
        content=mask.tobytes(),
    )
    assert response.status_code == 201, response.text
    before = service.store.get(Asset, asset["id"])
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_EXPORT_TEMP_BYTES", "128")
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "1")
    job = client.post(f"/api/projects/{setup['project_id']}/exports", {"asset_ids": [asset["id"]]})
    failed = wait_job(service.store, job["id"])
    assert failed.status == JobStatus.FAILED
    assert "storage is full" in failed.error
    assert failed.result == {}
    assert service.store.get(Asset, asset["id"]) == before
    reservations = TemporaryReservations(service.store.path.parent / "temporary-spools")
    assert held(reservations) == []
    assert not any(path.is_dir() for path in reservations.root.iterdir())
    assert http.get(f"/api/jobs/{job['id']}/export").status_code == 409
    monkeypatch.setenv("MONAILABEL_DEMO_EXPORT_TEMP_BYTES", str(1024**2))
    retry = client.post(
        f"/api/projects/{setup['project_id']}/exports", {"asset_ids": [asset["id"]]}
    )
    assert wait_job(service.store, retry["id"]).status == JobStatus.SUCCEEDED
    result = http.get(f"/api/jobs/{retry['id']}/export")
    assert result.status_code == 200
    with ZipFile(io.BytesIO(result.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["assets"][0]["revision"] == before.revision
        assert len(manifest["assets"][0]["annotations"]) == 1
    assert service.store.get(Asset, asset["id"]) == before
    assert held(reservations) == []
