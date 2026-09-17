"""A local durable job ledger with cooperative cancellation and atomic publication."""

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
)
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


def append_log(
    session: Session, job: Job, message: str, level: Literal["info", "error"] = "info"
) -> Job:
    job = job.model_copy(update={"log_count": job.log_count + 1})
    session.insert(
        JobLog(
            project_id=job.project_id,
            job_id=job.id,
            sequence=job.log_count,
            level=level,
            message=message[:4000],
        )
    )
    return job


@dataclass
class Outcome:
    result: dict[str, JsonValue]
    records: list[Record] = field(default_factory=list)


class JobContext:
    def __init__(self, store: Store, job_id: str):
        self.store, self.job_id = store, job_id

    def progress(self, value: float, message: str | None = None) -> None:
        with self.store.transaction() as session:
            job = session.get(Job, self.job_id)
            if job.status == JobStatus.CANCELLED:
                raise Cancelled()
            session.update(
                job.model_copy(
                    update={
                        "progress": min(0.99, max(0, value)),
                        "progress_message": job.progress_message if message is None else message,
                    }
                )
            )

    def log(self, message: str, level: Literal["info", "error"] = "info") -> None:
        with self.store.transaction() as session:
            job = session.get(Job, self.job_id)
            if job.status in TERMINAL_STATUSES:
                raise Cancelled()
            job = append_log(session, job, message, level)
            session.update(job.model_copy(update={"progress_message": message[:4000]}))


class Jobs:
    def __init__(self, store: Store, workers: int = 2):
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="monailabel")
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
        with self.store.transaction() as session:
            if key:
                for old in session.list(Job, project_id):
                    if old.idempotency_key == key:
                        if old.kind != kind or old.request != request:
                            raise Conflict(
                                "Idempotency key was already used for a different request."
                            )
                        return old
            if guard:
                guard(session)
            job = Job(project_id=project_id, kind=kind, request=request, idempotency_key=key)
            session.insert(job)
            if title := LOGGED_JOBS.get(kind):
                job = append_log(session, job, f"{title} queued.")
                session.update(job)
        self.executor.submit(self._run, job.id, work)
        return job

    def _run(self, identifier: str, work: Callable[[JobContext], Outcome]) -> None:
        try:
            with self.store.transaction() as session:
                job = session.get(Job, identifier)
                if job.status == JobStatus.CANCELLED:
                    return
                if title := LOGGED_JOBS.get(job.kind):
                    job = append_log(session, job, f"{title} started.")
                session.update(job.model_copy(update={"status": JobStatus.RUNNING}))
            outcome = work(JobContext(self.store, identifier))
            with self.store.transaction() as session:
                job = session.get(Job, identifier)
                if job.status == JobStatus.CANCELLED:
                    return
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
            return
        except Exception as exc:
            if not isinstance(exc, DomainError):
                logger.exception("Job %s failed", identifier)
            message = (
                str(exc) if isinstance(exc, DomainError) else "Job failed; inspect server logs."
            )
            with self.store.transaction() as session:
                try:
                    job = session.get(Job, identifier)
                except NotFound:
                    # A cancelled job's project/data may have been deleted while it unwound.
                    return
                if job.status != JobStatus.CANCELLED:
                    if job.kind in LOGGED_JOBS:
                        job = append_log(session, job, message, "error")
                    session.update(
                        job.model_copy(update={"status": JobStatus.FAILED, "error": message})
                    )

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
        self.executor.shutdown(wait=True, cancel_futures=False)
