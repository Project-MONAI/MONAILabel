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

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import test_demo_lifecycle
from test_temporary_storage import held, wait_job

from monailabel.core.errors import DomainError
from monailabel.core.models import JobStatus
from monailabel.server.demo_access import DemoAccess
from monailabel.server.demo_compute import isolated, write_result
from monailabel.server.demo_context import acting_as
from monailabel.server.jobs import Outcome
from monailabel.server.temporary_storage import TemporaryReservations

demo_http = test_demo_lifecycle.demo_http

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux isolated demo workers")


def test_oversized_serialized_result_is_replaced_by_bounded_error(tmp_path):
    write_result(tmp_path, b"x" * 10000, 1024)
    assert (tmp_path / "result").stat().st_size <= 1024
    with (tmp_path / "result").open("rb") as stream:
        error = pickle.load(stream)
    assert isinstance(error, DomainError)
    assert error.code == "storage_quota_exceeded" and error.status == 507


def test_worker_file_limit_is_inherited_and_never_changes_parent(tmp_path, monkeypatch):
    import resource

    before = resource.getrlimit(resource.RLIMIT_FSIZE), resource.getrlimit(resource.RLIMIT_CORE)
    monkeypatch.setenv("MONAILABEL_DEMO_COMPUTE_FILE_BYTES", "4096")
    child = """
import errno, resource, sys
from pathlib import Path
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (4096,4096)
assert resource.getrlimit(resource.RLIMIT_CORE) == (0,0)
path = Path(sys.argv[1])
try:
    path.write_bytes(b'x'*8192)
except OSError as error:
    assert error.errno == errno.EFBIG
else:
    raise AssertionError('File limit was bypassed')
assert path.stat().st_size <= 4096
"""
    script = """
import json, resource, subprocess, sys
from monailabel.server.demo_compute import worker_limits
worker_limits()
subprocess.run([sys.executable,'-c',sys.argv[1],sys.argv[2]],check=True,timeout=10)
print(json.dumps(resource.getrlimit(resource.RLIMIT_FSIZE)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, child, str(tmp_path / "bounded")],
        timeout=30,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == [4096, 4096]
    assert before == (
        resource.getrlimit(resource.RLIMIT_FSIZE),
        resource.getrlimit(resource.RLIMIT_CORE),
    )


@pytest.mark.parametrize("failure", ["result", "provider_file"])
def test_real_worker_main_returns_storage_error_without_publishing_result(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("MONAILABEL_DEMO_COMPUTE_FILE_BYTES", "4096")
    with (tmp_path / "input").open("wb") as stream:
        pickle.dump(("frame", ()), stream)
    script = """
import sys
from pathlib import Path
import monailabel.server.demo_compute as compute
directory = Path(sys.argv[1])
if sys.argv[2] == 'result':
    compute.execute = lambda *args: b'x'*8192
else:
    compute.execute = lambda *args: (directory/'provider-output').write_bytes(b'x'*8192)
sys.argv = ['worker', str(directory), '1024']
compute.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), failure], timeout=30, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
    assert (tmp_path / "result").stat().st_size <= 1024
    with (tmp_path / "result").open("rb") as stream:
        error = pickle.load(stream)
    assert isinstance(error, DomainError) and error.status == 507
    assert all(path.stat().st_size <= 4096 for path in tmp_path.iterdir())


def submit_inference(http, arguments=()):
    owner = test_demo_lifecycle.register(http)
    service = http.app.state.services
    project = service.store.get(DemoAccess, owner).initial_project_id

    def work(context):
        isolated("frame", *arguments)
        return Outcome({"must_not_publish": True})

    with acting_as(owner):
        job = service.jobs.submit("annotate", project, {}, work)
    return service, job


def assert_clean(service):
    reservations = TemporaryReservations(service.store.path.parent / "temporary-spools")
    assert held(reservations) == []
    assert not any(path.is_dir() for path in reservations.root.iterdir())


@pytest.mark.parametrize("failure", ["input", "configuration"])
def test_worker_refused_before_spawn_on_full_input_or_invalid_limit(
    demo_http, monkeypatch, failure
):
    monkeypatch.setenv("MONAILABEL_DEMO_COMPUTE_TEMP_BYTES", "1024")
    if failure == "configuration":
        monkeypatch.setenv("MONAILABEL_DEMO_COMPUTE_FILE_BYTES", "0")

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected inference must not start a child process")

    monkeypatch.setattr("monailabel.server.demo_compute.subprocess.Popen", forbidden)
    service, job = submit_inference(demo_http, (b"x" * 2048,) if failure == "input" else ())
    failed = wait_job(service.store, job.id)
    assert failed.status == JobStatus.FAILED
    assert failed.result == {}
    assert ("storage is full" if failure == "input" else "positive storage limits") in failed.error
    assert_clean(service)


def test_parent_checks_result_size_before_deserializing_and_discards_raw_logs(
    demo_http, monkeypatch
):
    monkeypatch.setenv("MONAILABEL_DEMO_COMPUTE_TEMP_BYTES", "1024")
    original = subprocess.Popen
    paths = []
    script = """
import sys
from pathlib import Path
(Path(sys.argv[1])/'result').write_bytes(b'not-a-pickle'+b'x'*int(sys.argv[2]))
print('x'*1000000)
"""

    def start(command, **kwargs):
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.STDOUT
        directory = Path(command[-2])
        paths.append(directory)
        assert Path(kwargs["env"]["TMPDIR"]) == directory / "provider-scratch"
        assert kwargs["env"]["TMP"] == kwargs["env"]["TEMP"] == kwargs["env"]["TMPDIR"]
        return original([sys.executable, "-c", script, *command[-2:]], **kwargs)

    monkeypatch.setattr("monailabel.server.demo_compute.subprocess.Popen", start)
    service, job = submit_inference(demo_http)
    failed = wait_job(service.store, job.id)
    assert failed.status == JobStatus.FAILED and "storage is full" in failed.error
    assert failed.result == {}
    assert paths and not any(path.exists() for path in paths)
    assert_clean(service)
