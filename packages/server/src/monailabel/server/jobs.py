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

"""A local durable job ledger with cooperative cancellation and atomic publication."""

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Literal

from pydantic import JsonValue

from monailabel.core.errors import Cancelled, Conflict, DomainError, NotFound
from monailabel.core.models import (
    TERMINAL_STATUSES,
    Asset,
    Job,
    JobLog,
    JobLogPage,
    JobStatus,
    Record,
    User,
)
from monailabel.server.demo_access import DemoBusy, access, lifecycle_enabled, require_active
from monailabel.server.demo_context import acting_as, actor
from monailabel.server.demo_policy import visitor
from monailabel.server.demo_training import require_live_training
from monailabel.server.evaluation_sets import EvaluationSets
from monailabel.server.storage import Session, Store

logger = logging.getLogger(__name__)
LOG_LIMIT = 1000
LOGGED_JOBS = {
    "train": "Training",
    "evaluate": "Evaluation",
    "training_report": "Evaluation",
    "batch_annotate": "Batch segmentation",
}
current_job: ContextVar["JobContext | None"] = ContextVar("monailabel_job", default=None)


def append_log(
    session: Session, job: Job, message: str, level: Literal["info", "error"] = "info"
) -> Job:
    try:
        session.insert(
            JobLog(
                project_id=job.project_id,
                job_id=job.id,
                sequence=job.log_count + 1,
                level=level,
                message=message[:4000],
            )
        )
    except DomainError as error:
        if error.code != "storage_quota_exceeded":
            raise
        # A full log allowance must not prevent cancellation/failure/expiry.
        # Existing log entries and sequence numbers remain unchanged.
        return job
    return job.model_copy(update={"log_count": job.log_count + 1})


@dataclass
class Outcome:
    result: dict[str, JsonValue]
    records: list[Record] = field(default_factory=list)


class JobContext:
    def __init__(self, store: Store, job_id: str):
        self.store, self.job_id = store, job_id

    def check_cancelled(self) -> None:
        job = self.store.get(Job, self.job_id)
        if job.status in TERMINAL_STATUSES:
            raise Cancelled()
        if job.user_id:
            try:
                require_active(self.store, self.store.get(User, job.user_id))
            except DomainError as error:
                raise Cancelled() from error

    def timed(self) -> bool:
        job = self.store.get(Job, self.job_id)
        return bool(lifecycle_enabled() and job.user_id and access(self.store, job.user_id))

    def progress(self, value: float, message: str | None = None) -> None:
        self.check_cancelled()
        with self.store.transaction() as session:
            job = session.get(Job, self.job_id)
            if job.status == JobStatus.CANCELLED:
                raise Cancelled()
            session.update(
                job.model_copy(
                    update={
                        "progress": min(0.99, max(0, value)),
                        "progress_message": job.progress_message
                        if message is None
                        else message[:4000],
                    }
                )
            )

    def log(self, message: str, level: Literal["info", "error"] = "info") -> None:
        self.check_cancelled()
        with self.store.transaction() as session:
            job = session.get(Job, self.job_id)
            if job.status in TERMINAL_STATUSES:
                raise Cancelled()
            job = append_log(session, job, message, level)
            session.update(job.model_copy(update={"progress_message": message[:4000]}))


class Jobs:
    def __init__(self, store: Store, workers: int = 2, *, capacity: int = 32, per_user: int = 4):
        if capacity < 1 or per_user < 1:
            raise ValueError("Job capacities must be positive.")
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="monailabel")
        self.running_lock = Lock()
        self.running: set[str] = set()
        self.submission_lock = RLock()
        self.pending: dict[str, str | None] = {}
        self.capacity, self.per_user = capacity, per_user
        self.closed = False
        with store.transaction() as session:
            for job in session.list(Job):
                if job.status not in TERMINAL_STATUSES:
                    if title := LOGGED_JOBS.get(job.kind):
                        job = append_log(
                            session, job, f"{title} interrupted by server restart.", "error"
                        )
                    session.update(
                        job.model_copy(
                            update={
                                "status": JobStatus.INTERRUPTED,
                                "error": "Server restarted. Submit a new job to retry.",
                            }
                        )
                    )

    def submit(
        self,
        kind: str,
        project_id: str,
        request: dict[str, JsonValue],
        work: Callable[[JobContext], Outcome],
        key: str | None = None,
        *,
        guard: Callable[[Session], None] | None = None,
    ) -> Job:
        # Serialize reservation with ledger publication and executor submission.
        # The executor's queue is otherwise unbounded. Cancelled queued futures
        # keep their reservation until consumed, so cancel/retry cannot fill it.
        with self.submission_lock:
            if self.closed:
                raise DomainError("The job service is shutting down.", status=503)
            return self._submit(kind, project_id, request, work, key, guard=guard)

    def _submit(
        self,
        kind: str,
        project_id: str,
        request: dict[str, JsonValue],
        work: Callable[[JobContext], Outcome],
        key: str | None = None,
        *,
        guard: Callable[[Session], None] | None = None,
    ) -> Job:
        if kind == "train":
            require_live_training()
        with self.store.transaction() as session:
            user_id = actor.get()
            if user_id:
                require_active(session, session.get(User, user_id))
                if access(session, user_id) and kind in {
                    "train",
                    "classify",
                    "roi",
                    "bounding_box",
                }:
                    raise DomainError(
                        "Live training and hosted annotation are unavailable "
                        "in this timed local demo.",
                        code="demo_restricted",
                        status=403,
                    )
            if key:
                for old in session.list(Job, project_id):
                    if old.idempotency_key == key:
                        if old.kind != kind or old.request != request:
                            raise Conflict(
                                "Idempotency key was already used for a different request."
                            )
                        return old
            if (
                len(self.pending) >= self.capacity
                or sum(owner == user_id for owner in self.pending.values()) >= self.per_user
            ):
                raise DemoBusy(
                    "The job queue is full for this account or server. "
                    "Wait for existing work to finish before submitting another job.",
                    5,
                    code="job_queue_full",
                )
            if guard:
                guard(session)
            job = Job(
                project_id=project_id,
                user_id=user_id,
                kind=kind,
                request=request,
                idempotency_key=key,
            )
            session.insert(job)
            if title := LOGGED_JOBS.get(kind):
                job = append_log(session, job, f"{title} queued.")
                session.update(job)
        self.pending[job.id] = user_id
        try:
            future = self.executor.submit(self._run, job.id, work)
            future.add_done_callback(lambda _: self._release(job.id))
        except RuntimeError as exc:
            self._release(job.id)
            with self.store.transaction() as session:
                session.update(
                    job.model_copy(
                        update={"status": JobStatus.FAILED, "error": "Job could not be scheduled."}
                    )
                )
            raise DomainError("Job could not be scheduled. Try again later.", status=503) from exc
        return job

    def _release(self, identifier: str) -> None:
        with self.submission_lock:
            self.pending.pop(identifier, None)

    def _run(self, identifier: str, work: Callable[[JobContext], Outcome]) -> None:
        with self.running_lock:
            self.running.add(identifier)
        try:
            context = JobContext(self.store, identifier)
            context.check_cancelled()
            with self.store.transaction() as session:
                job = session.get(Job, identifier)
                if job.status == JobStatus.CANCELLED:
                    return
                if title := LOGGED_JOBS.get(job.kind):
                    job = append_log(session, job, f"{title} started.")
                session.update(job.model_copy(update={"status": JobStatus.RUNNING}))
            token = current_job.set(context)
            try:
                with acting_as(
                    job.user_id,
                    prepared_only=bool(job.user_id and visitor(self.store, job.user_id)),
                ):
                    outcome = work(context)
                    context.check_cancelled()
            finally:
                current_job.reset(token)
            with acting_as(job.user_id), self.store.transaction() as session:
                job = session.get(Job, identifier)
                if job.status == JobStatus.CANCELLED:
                    return
                if job.user_id:
                    require_active(session, session.get(User, job.user_id))
                for record in outcome.records:
                    session.insert(record)
                    if isinstance(record, Asset):
                        EvaluationSets.on_import(session, record, is_new=True)
                if title := LOGGED_JOBS.get(job.kind):
                    suffix = (
                        " Model saved."
                        if job.kind == "train"
                        else " Report saved."
                        if job.kind in {"evaluate", "training_report"}
                        else ""
                    )
                    job = append_log(session, job, f"{title} completed.{suffix}")
                session.update(
                    job.model_copy(
                        update={
                            "status": JobStatus.SUCCEEDED,
                            "progress": 1.0,
                            "result": outcome.result,
                        }
                    )
                )
        except Cancelled:
            self.cancel(identifier)
            return
        except Exception as exc:
            if not isinstance(exc, DomainError):
                logger.exception("Job %s failed", identifier)
            message = (
                str(exc) if isinstance(exc, DomainError) else "Job failed; inspect server logs."
            )[:4000]
            with self.store.transaction() as session:
                try:
                    job = session.get(Job, identifier)
                except NotFound:
                    # A cancelled job's project/data may have been deleted while it unwound.
                    return
                if job.status != JobStatus.CANCELLED:
                    if job.user_id:
                        try:
                            require_active(session, session.get(User, job.user_id))
                        except DomainError:
                            session.update(job.model_copy(update={"status": JobStatus.CANCELLED}))
                            return
                    if job.kind in LOGGED_JOBS:
                        job = append_log(session, job, message, "error")
                    session.update(
                        job.model_copy(update={"status": JobStatus.FAILED, "error": message})
                    )
        finally:
            with self.running_lock:
                self.running.discard(identifier)

    def executing(self) -> set[str]:
        with self.running_lock:
            return set(self.running)

    def cancel(self, identifier: str) -> Job:
        with self.store.transaction() as session:
            job = session.get(Job, identifier)
            if job.status not in TERMINAL_STATUSES:
                if title := LOGGED_JOBS.get(job.kind):
                    job = append_log(session, job, f"{title} cancelled.")
                job = job.model_copy(update={"status": JobStatus.CANCELLED})
                session.update(job)
        return job

    def logs(self, identifier: str, after: int = 0, limit: int | None = None) -> JobLogPage:
        limit = limit or LOG_LIMIT
        with self.store.transaction() as session:
            job = session.get(Job, identifier)
            rows = session.connection.execute(
                "SELECT data FROM records WHERE kind='JobLog' AND project_id=? "
                "AND json_extract(data, '$.job_id')=? AND json_extract(data, '$.sequence')>? "
                "ORDER BY json_extract(data, '$.sequence') DESC LIMIT ?",
                (job.project_id, identifier, after, limit),
            )
            return JobLogPage(
                entries=list(reversed([JobLog.model_validate_json(row[0]) for row in rows])),
                next_cursor=job.log_count,
                truncated=after < job.log_count - limit,
                total_lines=job.log_count,
            )

    def download_log(self, identifier: str) -> Iterator[str]:
        # Capture the end once: downloading an active run must still finish.
        job = self.store.get(Job, identifier)
        after = 0
        while True:
            with self.store.transaction() as session:
                rows = session.connection.execute(
                    "SELECT data FROM records WHERE kind='JobLog' AND project_id=? "
                    "AND json_extract(data, '$.job_id')=? "
                    "AND json_extract(data, '$.sequence')>? "
                    "AND json_extract(data, '$.sequence')<=? "
                    "ORDER BY json_extract(data, '$.sequence') LIMIT ?",
                    (job.project_id, identifier, after, job.log_count, LOG_LIMIT),
                )
                entries = [JobLog.model_validate_json(row[0]) for row in rows]
            if not entries:
                return
            for entry in entries:
                after = entry.sequence
                yield f"{entry.created_at.isoformat()} {entry.level.upper()} {entry.message}\n"

    def close(self) -> None:
        with self.submission_lock:
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=False)
