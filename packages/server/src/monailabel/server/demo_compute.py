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

"""Killable inference for timed Linux demo visits; not a security sandbox.

Only server-constructed arguments enter the private temporary pickle files. No
request can select a callable, supply a pickle, or choose a worker file path.
Workers use a fresh interpreter (never fork a CUDA-initialized server).
Cancelling a hosted request stops local work, not an upstream charge already incurred.
"""

import errno
import os
import pickle
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from monailabel.core.errors import DomainError
from monailabel.server.demo_gpu import gpu_slot
from monailabel.server.storage_quota import exceeded, limit
from monailabel.server.temporary_storage import Spool, job_spool

Operation = Literal["predict", "prompted", "track", "frame"]


def timed_compute() -> bool:
    from monailabel.server.jobs import current_job

    context = current_job.get()
    return context is not None and context.timed()


def isolated(operation: Operation, *arguments: Any) -> Any:
    from monailabel.server.jobs import current_job

    context = current_job.get()
    if context is None or not context.timed():
        raise DomainError("Isolated demo compute requires a timed job.")
    if sys.platform != "linux":
        raise DomainError("Timed demo compute currently requires Linux.")
    context.check_cancelled()
    with (
        gpu_slot(context.check_cancelled, enabled=operation != "frame") as device,
        job_spool(context, "COMPUTE_TEMP_BYTES", 4 * 1024**3) as spool,
    ):
        directory = spool.directory
        with spool.open("input") as stream:
            pickle.dump((operation, arguments), stream, protocol=5)
        # Timed inference always belongs to a persisted visitor lease.
        assert spool.capacity is not None
        remaining = spool.capacity - spool.sizes["input"]
        if remaining <= 0:
            raise exceeded()
        scratch = directory / "provider-scratch"
        scratch.mkdir(mode=0o700)
        context.check_cancelled()
        environment = os.environ | {
            "MONAILABEL_DEMO_WORKER": "1",
            "MONAILABEL_PREPARED_MODELS_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "TMPDIR": str(scratch),
            "TEMP": str(scratch),
            "TMP": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MONAILABEL_DEMO_COMPUTE_FILE_BYTES": str(limit("COMPUTE_FILE_BYTES", 2 * 1024**3)),
        }
        if device is not None:
            # Set visibility only in the new child, never in the threaded server.
            # Its selected physical device is CUDA device zero for every adapter.
            environment["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(directory), str(remaining)],
            env=environment,
            # Raw provider output was never surfaced and could fill worker.log.
            # Structured domain errors still return through the bounded result.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                context.check_cancelled()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.1)
            context.check_cancelled()
            result = directory / "result"
            if process.returncode or not result.is_file():
                raise DomainError("The isolated demo inference worker failed.")
            if result.stat().st_size > remaining:
                raise exceeded()
            with result.open("rb") as stream:
                value = pickle.load(stream)
            if isinstance(value, DomainError):
                raise value
            context.check_cancelled()
            return value
        finally:
            # Kill the private group, including model/ffmpeg subprocesses, before
            # removing scratch or releasing its input/result reservation.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


class TimedSpatial:
    def predict_prompted(self, *args: Any) -> Any:
        # Last argument is the parent's progress callback, never serialized.
        return isolated("prompted", *args[:-1])


def execute(operation: Operation, arguments: tuple[Any, ...]) -> Any:
    if operation == "predict":
        from monailabel.server.models import Models
        from monailabel.server.storage import Artifacts, Store

        database, artifacts, project, model, image, prompt, affine = arguments

        def no_credentials(project_id: str, identifier: str) -> str:
            raise DomainError("Visitor workers cannot resolve project credentials.")

        return Models(Store(database), Artifacts(artifacts), no_credentials).predict(
            project, model, image, prompt, affine
        )
    if operation == "prompted":
        from monailabel.server.models import Models

        image, label, model, spatial, plane, full = arguments
        return Models.spatial_provider(model).predict_prompted(
            image, label, model, spatial, plane, full, lambda value: None
        )
    if operation == "track":
        from monailabel.sam.video import SamVideoTracker

        source, width, height, seed, count, output, mask = arguments
        return SamVideoTracker().track(
            source, width, height, seed, count, lambda value: None, output, mask
        )
    if operation == "frame":
        from types import SimpleNamespace

        from monailabel.server.storage import Artifacts
        from monailabel.server.video.tracking import VideoTracking

        root, asset, frame = arguments
        return VideoTracking.frame(SimpleNamespace(artifacts=Artifacts(root)), asset, frame)  # type: ignore[arg-type]
    raise DomainError("Unknown isolated demo operation.")


def worker_limits() -> int:
    # Set limits only in the fresh Linux child, never the shared server process.
    # Descendants inherit the hard file limit; this is not an aggregate disk quota.
    import resource

    previous = resource.getrlimit(resource.RLIMIT_FSIZE)
    ceiling = min(
        [
            limit("COMPUTE_FILE_BYTES", 2 * 1024**3),
            *(bound for bound in previous if bound != resource.RLIM_INFINITY),
        ]
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (ceiling, ceiling))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return ceiling


def write_result(directory: Path, value: Any, capacity: int) -> None:
    spool = Spool(directory, capacity)
    try:
        with spool.open("result") as stream:
            pickle.dump(value, stream, protocol=5)
    except DomainError:
        # Replace only our incomplete disposable result, never any saved work.
        spool.remove("result")
        with spool.open("result") as stream:
            pickle.dump(exceeded(), stream, protocol=5)


def main() -> None:
    directory = Path(sys.argv[1])
    capacity = int(sys.argv[2])
    if not 0 < capacity < 2**63:
        raise ValueError("Invalid worker result capacity")
    capacity = min(capacity, worker_limits())
    with (directory / "input").open("rb") as stream:
        operation, arguments = pickle.load(stream)
    try:
        value = execute(operation, arguments)
    except DomainError as error:
        value = error
    except OSError as error:
        if error.errno not in {errno.EFBIG, errno.ENOSPC, errno.EDQUOT}:
            raise
        value = exceeded()
    write_result(directory, value, capacity)


if __name__ == "__main__":
    main()
