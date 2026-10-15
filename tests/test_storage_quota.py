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

"""All quota writes, crashes and recovery run in disposable pytest workspaces."""

import hashlib
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, Job, JobStatus, Membership, Project, Role, User
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_context import acting_as
from monailabel.server.jobs import Jobs, Outcome
from monailabel.server.storage import Artifacts, Store


@pytest.fixture
def storage(tmp_path, monkeypatch):
    for suffix in ("SELF_REGISTER", "LIFECYCLE", "STORAGE_LIMITS"):
        monkeypatch.setenv("MONAILABEL_DEMO_" + suffix, "0")
    monkeypatch.setattr(
        "monailabel.server.storage_quota.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1024**4),
    )
    store = Store(tmp_path / "workspace.sqlite3")
    with store.transaction() as session:
        for identifier in ("first", "second"):
            session.insert(User(id=identifier, username=identifier))
            session.insert(DemoAccess(id=identifier, minutes=60))
        session.insert(User(id="operator", username="operator"))
    artifacts = Artifacts(tmp_path / "artifacts", database=store.path)
    return store, artifacts


def usage(artifacts, owner=None):
    with artifacts.quota.connection() as connection:
        return artifacts.quota.usage(connection, owner)


def full(error):
    assert error.value.status == 507
    assert error.value.code == "storage_quota_exceeded"


def test_visitor_bytes_are_cumulative_durable_and_deduplicated(storage, monkeypatch):
    store, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "8")
    with acting_as("first"):
        key = artifacts.put(b"12345678")
        assert artifacts.put(b"12345678") == key
        with pytest.raises(DomainError) as error:
            artifacts.put(b"9")
        full(error)
    assert usage(artifacts, "first") == (8, 1)
    reopened = Artifacts(artifacts.root, database=store.path)
    reopened.quota.reconcile()
    with acting_as("first"), pytest.raises(DomainError) as error:
        reopened.put(b"9")
    full(error)
    assert artifacts.read(key) == b"12345678"
    assert list(artifacts.root.rglob(".pending-*")) == []


@pytest.mark.parametrize("setting", ["ARTIFACT_OBJECTS", "ARTIFACT_TOTAL_OBJECTS"])
def test_object_limits_bound_empty_and_small_blobs(storage, monkeypatch, setting):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_" + setting, "1")
    with acting_as("first"):
        key = artifacts.put(b"")
        with pytest.raises(DomainError) as error:
            artifacts.put(b"x")
        full(error)
        assert artifacts.put(b"") == key
    assert usage(artifacts) == (0, 1)


def test_shared_files_count_once_physically_but_for_each_visitor(storage, monkeypatch):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "4")
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_BYTES", "4")
    host_key = artifacts.put(b"host")
    for owner in ("first", "second"):
        with acting_as(owner):
            assert artifacts.put(b"host") == host_key
            with pytest.raises(DomainError) as error:
                artifacts.put(b"x")
            full(error)
        assert usage(artifacts, owner) == (4, 1)
    assert usage(artifacts) == (4, 1)


def test_shared_file_cannot_bypass_smaller_visitor_budget(storage, monkeypatch):
    _, artifacts = storage
    key = artifacts.put(b"prepared host data")
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "1")
    with acting_as("first"), pytest.raises(DomainError) as error:
        artifacts.put(b"prepared host data")
    full(error)
    assert usage(artifacts, "first") == (0, 0)
    assert artifacts.read(key) == b"prepared host data"


def test_claim_limit_bounds_cross_visitor_deduplication(storage, monkeypatch):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_CLAIMS", "1")
    with acting_as("first"):
        key = artifacts.put(b"shared")
    with acting_as("second"), pytest.raises(DomainError) as error:
        artifacts.put(b"shared")
    full(error)
    with acting_as("first"):
        assert artifacts.put(b"shared") == key
    assert usage(artifacts, "second") == (0, 0)


@pytest.mark.parametrize("owner", [None, "operator"])
def test_operator_behavior_and_optional_global_enforcement(storage, monkeypatch, owner):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_BYTES", "1")
    with acting_as(owner):
        key = artifacts.put(b"operator data")
        monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_LIMITS", "1")
        with pytest.raises(DomainError) as error:
            artifacts.put(b"new")
        full(error)
        assert artifacts.put(b"operator data") == key


@pytest.mark.parametrize("value", ["0", "-1", "no limit", str(2**63)])
def test_bad_host_limit_fails_closed(storage, monkeypatch, value):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", value)
    with acting_as("first"), pytest.raises(DomainError) as error:
        artifacts.put(b"data")
    assert error.value.status == 503
    assert usage(artifacts) == (0, 0)


def test_free_space_headroom_checked_before_spool(storage, monkeypatch):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_STORAGE_FREE_BYTES", "10")
    monkeypatch.setattr(
        "monailabel.server.storage_quota.shutil.disk_usage",
        lambda path: SimpleNamespace(free=13),
    )
    before = set(artifacts.root.iterdir())
    with acting_as("first"), pytest.raises(DomainError) as error:
        artifacts.put(b"four")
    full(error)
    assert set(artifacts.root.iterdir()) == before
    assert usage(artifacts) == (0, 0)


@pytest.mark.parametrize("layout", ["C", "F", "sliced", "scalar", "empty", "structured"])
def test_array_budget_includes_exact_numpy_encoding(storage, monkeypatch, layout):
    _, artifacts = storage
    array = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    if layout == "F":
        array = np.asfortranarray(array)
    elif layout == "sliced":
        array = array[:, ::2, ::-1]
    elif layout == "scalar":
        array = np.asarray(7, dtype=np.float64)
    elif layout == "empty":
        array = np.empty((0, 3), dtype=np.uint8)
    elif layout == "structured":
        array = np.zeros(2, dtype=[("point", "<f8", (3,)), ("label", "u1")])
    encoded = io.BytesIO()
    np.save(encoded, array, allow_pickle=False)
    content = encoded.getvalue()
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", str(len(content) - 1))
    with acting_as("first"):
        with pytest.raises(DomainError) as error:
            artifacts.put_array(array)
        full(error)
        assert usage(artifacts) == (0, 0)
        monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", str(len(content)))
        key = artifacts.put_array(array)
    assert key == hashlib.sha256(content).hexdigest() + ".npy"
    assert artifacts.read(key) == content
    np.testing.assert_array_equal(artifacts.array(key), array)
    assert usage(artifacts, "first") == (len(content), 1)


def test_streamed_files_and_json_share_the_same_budget(storage, monkeypatch, tmp_path):
    _, artifacts = storage
    source = tmp_path / "disposable-upload"
    source.write_bytes(b"12345")
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "13")
    with acting_as("first"):
        key = artifacts.put_file(source)
        manifest = artifacts.put_json({"x": 1})
        with pytest.raises(DomainError) as error:
            artifacts.put_json({"x": 2})
        full(error)
    assert artifacts.read(key) == source.read_bytes()
    assert artifacts.json(manifest) == {"x": 1}
    assert usage(artifacts, "first") == (13, 2)


def test_changed_stream_source_is_never_published(storage, monkeypatch, tmp_path):
    _, artifacts = storage
    source = tmp_path / "disposable-changing-file"
    source.write_bytes(b"same size original")
    original = artifacts._put

    def mutate(key, size, write):
        source.write_bytes(b"same size modified")
        return original(key, size, write)

    monkeypatch.setattr(artifacts, "_put", mutate)
    with acting_as("first"), pytest.raises(DomainError, match="changed while being saved"):
        artifacts.put_file(source)
    assert usage(artifacts) == (0, 0)
    assert list(artifacts.quota.files()) == []
    assert list(artifacts.root.rglob(".pending-*")) == []


def test_late_publication_error_keeps_published_file_charged(storage, monkeypatch):
    _, artifacts = storage
    calls = 0
    original = os.fsync

    def fail_directory(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated directory sync failure after publication")
        return original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_directory)
        with acting_as("first"), pytest.raises(OSError):
            artifacts.put(b"published")
    assert usage(artifacts, "first") == (9, 1)
    with acting_as("first"):
        key = artifacts.put(b"published")
    assert artifacts.read(key) == b"published"
    assert usage(artifacts) == (9, 1)


@pytest.mark.parametrize("mutation", ["larger", "smaller", "different", "exception"])
def test_failed_or_changed_write_releases_only_removed_partial(storage, mutation):
    _, artifacts = storage
    key = hashlib.sha256(b"four").hexdigest() + ".bin"

    def write(sink):
        sink.write(b"f")
        if mutation == "exception":
            raise OSError("simulated writer failure")
        sink.write({"larger": b"ive!!", "smaller": b"o", "different": b"ive"}[mutation])

    with acting_as("first"), pytest.raises(DomainError if mutation != "exception" else OSError):
        artifacts._put(key, 4, write)
    assert not artifacts.path(key).exists()
    assert list(artifacts.root.rglob(".pending-*")) == []
    assert usage(artifacts, "first") == usage(artifacts) == (0, 0)


def test_failed_partial_cleanup_keeps_reservation(storage, monkeypatch):
    _, artifacts = storage
    key = hashlib.sha256(b"four").hexdigest() + ".bin"
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name == ".pending-" + key:
            raise OSError("simulated cleanup failure")
        return original(path, *args, **kwargs)

    def write(sink):
        sink.write(b"f")
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        with acting_as("first"), pytest.raises(OSError):
            artifacts._put(key, 4, write)
    assert usage(artifacts, "first") == (4, 1)
    assert artifacts.path(key).with_name(".pending-" + key).read_bytes() == b"f"
    artifacts.quota.reconcile()
    assert usage(artifacts, "first") == usage(artifacts) == (0, 0)


def test_parallel_threads_cannot_overspend_shared_reservation(storage, monkeypatch):
    _, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "8")

    def write(index):
        with acting_as("first"):
            try:
                return artifacts.put(bytes([index]) * 4)
            except DomainError as error:
                assert error.status == 507
                return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(pool.map(write, range(16)))
    assert sum(key is not None for key in keys) == 2
    assert usage(artifacts, "first") == usage(artifacts) == (8, 2)


def test_parallel_processes_share_the_same_budget(storage, monkeypatch):
    store, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "8")
    program = """
import sys
from pathlib import Path
from monailabel.server.storage import Artifacts
from monailabel.server.demo_context import acting_as
from monailabel.core.errors import DomainError
artifacts = Artifacts(Path(sys.argv[1]), database=Path(sys.argv[2]))
with acting_as('first'):
    try:
        artifacts.put(bytes([int(sys.argv[3])]) * 4)
    except DomainError as error:
        assert error.status == 507
        sys.exit(7)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(artifacts.root), str(store.path), str(index)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(4)
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
    assert usage(artifacts, "first") == usage(artifacts) == (8, 2)


@pytest.mark.parametrize("published", [False, True])
def test_process_crash_reconciliation_preserves_published_data_and_other_drafts(storage, published):
    store, artifacts = storage
    key = hashlib.sha256(b"four").hexdigest() + ".bin"
    program = """
import os, sys
from pathlib import Path
from monailabel.server.storage import Artifacts
from monailabel.server.demo_context import acting_as
artifacts = Artifacts(Path(sys.argv[1]), database=Path(sys.argv[2]))
target = artifacts.path(sys.argv[3])
with acting_as('first'), artifacts.quota.reserve(target, 4) as temporary:
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(b'four')
    if sys.argv[4] == 'True':
        os.replace(temporary, target)
    os._exit(23)
"""
    process = subprocess.run(
        [sys.executable, "-c", program, str(artifacts.root), str(store.path), key, str(published)],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 23, process.stderr.decode()
    assert usage(artifacts, "first") == (4, 1)
    draft = artifacts.path(key).with_name("unrelated-draft.npy")
    draft.write_bytes(b"keep native draft")
    # Startup reconciliation only: no concurrent server/workers own this test workspace.
    restarted = Artifacts(artifacts.root, database=store.path)
    restarted.quota.reconcile()
    assert usage(restarted, "first") == ((4, 1) if published else (0, 0))
    assert draft.read_bytes() == b"keep native draft"
    assert list(artifacts.root.rglob(".pending-*")) == []
    if published:
        assert restarted.read(key) == b"four"


def test_startup_backfills_existing_nested_artifacts_without_deleting_them(storage, monkeypatch):
    store, artifacts = storage
    image = artifacts.put(b"old prepared data")
    nested = artifacts.put_json({"image": image})
    manifest = artifacts.put_json({"nested": nested, "same": [image]})
    project = Project(name="Disposable old workspace")
    asset = Asset(
        project_id=project.id,
        name="synthetic",
        group_id="synthetic",
        split="pool",
        image_key=manifest,
        spatial_shape=[2, 2],
    )
    with store.transaction() as session:
        session.insert(project)
        session.insert(asset)
        for owner in ("first", "second"):
            session.insert(Membership(project_id=project.id, user_id=owner, roles=[Role.ANNOTATOR]))
    assert usage(artifacts, "first") == (0, 0)
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_CLAIMS", "1")
    with pytest.raises(DomainError, match="increase it before startup") as error:
        artifacts.quota.reconcile()
    assert error.value.status == 503
    assert usage(artifacts, "first") == usage(artifacts, "second") == (0, 0)
    assert artifacts.read(image) == b"old prepared data"
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_CLAIMS", "6")
    artifacts.quota.reconcile()
    assert usage(artifacts, "first") == usage(artifacts, "second") == usage(artifacts)
    assert usage(artifacts)[1] == 3
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "1")
    with acting_as("first"), pytest.raises(DomainError) as error:
        artifacts.put(b"x")
    full(error)
    assert artifacts.read(image) == b"old prepared data"


def test_artifact_write_inside_workspace_transaction_does_not_deadlock(storage):
    store, artifacts = storage
    with acting_as("first"), store.transaction() as session:
        session.insert(Project(name="Disposable publication"))
        key = artifacts.put(b"published in transaction")
    assert artifacts.read(key) == b"published in transaction"
    assert usage(artifacts, "first") == (24, 1)


def test_legacy_host_artifacts_are_counted_at_first_initialization(storage, monkeypatch):
    store, _ = storage
    legacy = Artifacts(store.path.parent / "legacy-artifacts")
    key = legacy.put(b"older operator data")
    bounded = Artifacts(legacy.root, database=store.path)
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_TOTAL_BYTES", "1")
    with acting_as("first"), pytest.raises(DomainError) as error:
        bounded.put(b"new")
    full(error)
    assert bounded.read(key) == b"older operator data"
    assert usage(bounded) == (19, 1)


def test_background_job_rehydrates_visitor_storage_identity(storage, monkeypatch):
    store, artifacts = storage
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "4")
    jobs = Jobs(store, workers=1)
    try:
        with acting_as("first"):
            first = jobs.submit("test", "p", {}, lambda _: Outcome({"key": artifacts.put(b"four")}))
            second = jobs.submit(
                "test", "p", {}, lambda _: Outcome({"key": artifacts.put(b"five")})
            )
    finally:
        jobs.close()
    assert store.get(Job, first.id).status == JobStatus.SUCCEEDED
    failed = store.get(Job, second.id)
    assert failed.status == JobStatus.FAILED and "storage is full" in failed.error
    assert usage(artifacts, "first") == (4, 1)


def test_denied_review_preserves_revision_mask_and_decisions(http, client, seeded, monkeypatch):
    setup, assets = seeded
    asset = assets[0]
    services = http.app.state.services
    mask = np.asarray(client.get(f"/api/assets/{asset['id']}/fixture")["mask"], dtype=np.uint8)
    saved = client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask.tolist(), "covered_labels": [0, 1, 2]},
    )
    visitor = services.auth.create_user("quota-visitor", "test-password-1234", demo_minutes=60)
    services.auth.add_member(setup["project_id"], visitor.id, [Role.ANNOTATOR, Role.REVIEWER])
    client.post("/api/auth/login", {"username": visitor.username, "password": "test-password-1234"})
    monkeypatch.setenv("MONAILABEL_DEMO_ARTIFACT_BYTES", "1")
    changed = mask.copy()
    changed.flat[0] = (int(changed.flat[0]) + 1) % 3
    response = http.post(
        f"/api/assets/{asset['id']}/review",
        json={"base_revision": 1, "mask": changed.tolist(), "covered_labels": [0, 1, 2]},
    )
    assert response.status_code == 507, response.text
    assert response.json()["code"] == "storage_quota_exceeded"
    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
    assert client.get(f"/api/assets/{asset['id']}/annotations") == [saved]
    assert client.get(f"/api/projects/{setup['project_id']}/decisions") == []
    assert http.get(f"/api/annotations/{saved['id']}/mask.bin").content == mask.tobytes()
