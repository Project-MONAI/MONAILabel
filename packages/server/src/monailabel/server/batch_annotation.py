"""Batch inference with per-case publication and pending human review."""

import logging

from pydantic import JsonValue

from monailabel.core.errors import Cancelled, Conflict, DomainError
from monailabel.core.models import (
    TERMINAL_STATUSES,
    AnnotateRequest,
    Annotation,
    Asset,
    BatchAnnotateRequest,
    Job,
    JobStatus,
    Proposal,
    Split,
)
from monailabel.server.annotation import Annotations
from monailabel.server.evaluation_sets import reserved
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.storage import Session, Store, referenced_assets

logger = logging.getLogger(__name__)


def candidates(
    session: Session, project_id: str, *, current_job_id: str | None = None
) -> list[Asset]:
    groups, images = reserved(session, project_id)
    assets = session.list(Asset, project_id)
    for asset in assets:
        if asset.split == Split.VALIDATION:
            groups.add(asset.group_id)
            images.add(asset.image_key)
    busy = {p.asset_id for p in session.list(Proposal, project_id) if p.status == "pending"}
    for job in session.list(Job, project_id):
        if (
            job.id != current_job_id
            and job.status not in TERMINAL_STATUSES
            and job.kind in {"annotate", "batch_annotate"}
        ):
            busy.update(referenced_assets(job.request))
    return [
        a
        for a in assets
        if a.split != Split.VALIDATION
        and a.group_id not in groups
        and a.image_key not in images
        and a.annotation_id is None
        and a.id not in busy
    ]


class CaseContext(JobContext):
    def __init__(self, parent: JobContext, index: int, total: int, name: str):
        super().__init__(parent.store, parent.job_id)
        self.index, self.total, self.name = index, total, name

    def progress(self, value: float, message: str | None = None) -> None:
        super().progress(
            (self.index + value) / self.total,
            f"Image {self.index + 1} of {self.total} · {self.name}"
            + (f" · {message}" if message else ""),
        )


class BatchAnnotations:
    def __init__(self, store: Store, annotations: Annotations, jobs: Jobs):
        self.store, self.annotations, self.jobs = store, annotations, jobs

    def start(self, project_id: str, request: BatchAnnotateRequest, user_id: str) -> Job:
        with self.store.transaction() as session:
            assets = candidates(session, project_id)[: request.limit]
        if not assets:
            raise DomainError("No unannotated images are available outside evaluation-only data.")
        # Validate every case/model before starting any inference. Reuse the single-image path.
        operations = [
            self.annotations.prepare(
                asset.id,
                AnnotateRequest(
                    model_id=request.model_id,
                    label_ids=request.label_ids,
                    prompt=request.prompt,
                ),
            )
            for asset in assets
        ]
        asset_ids = [a.id for a in assets]

        def guard(session: Session) -> None:
            eligible = {a.id: a for a in candidates(session, project_id)}
            if any(a.id not in eligible or eligible[a.id].revision != a.revision for a in assets):
                raise Conflict("The available images changed. Retry the batch request.")

        def work(context: JobContext) -> Outcome:
            result: dict[str, JsonValue] = {
                "asset_ids": [],
                "proposal_ids": [],
                "annotation_ids": [],
                "failed": [],
                "selected_count": len(assets),
                "requested_count": request.limit,
                "submit_for_review": request.submit_for_review,
            }
            for index, (asset, operation) in enumerate(zip(assets, operations, strict=True)):
                case = CaseContext(context, index, len(assets), asset.name)
                case.progress(0)
                context.log(
                    f"Image {index + 1} of {len(assets)} · {asset.name}: segmentation started."
                )
                try:
                    outcome = operation(case)
                    proposal = next(r for r in outcome.records if isinstance(r, Proposal))
                    with self.store.transaction() as session:
                        job = session.get(Job, context.job_id)
                        if job.status == JobStatus.CANCELLED:
                            raise Cancelled()
                        current = session.get(Asset, asset.id)
                        eligible = {
                            a.id for a in candidates(session, project_id, current_job_id=job.id)
                        }
                        if current.revision != proposal.base_revision or current.id not in eligible:
                            raise Conflict(
                                "Image changed during segmentation; no annotation saved."
                            )
                        session.insert(proposal)
                        if request.submit_for_review:
                            annotation = Annotation(
                                project_id=project_id,
                                asset_id=asset.id,
                                revision=current.revision + 1,
                                mask_key=proposal.mask_key,
                                covered_labels=[0, *proposal.label_ids],
                                reviewer=user_id,
                                proposal_id=proposal.id,
                            )
                            session.insert(annotation)
                            session.update(
                                current.model_copy(
                                    update={
                                        "revision": annotation.revision,
                                        "annotation_id": annotation.id,
                                    }
                                )
                            )
                            # Applied proposal is distinct from a Good review decision.
                            session.update(proposal.model_copy(update={"status": "accepted"}))
                            ids = result["annotation_ids"]
                            assert isinstance(ids, list)
                            ids.append(annotation.id)
                        for key, value in (("asset_ids", asset.id), ("proposal_ids", proposal.id)):
                            ids = result[key]
                            assert isinstance(ids, list)
                            ids.append(value)
                        session.update(job.model_copy(update={"result": result}))
                    saved = (
                        "submitted for pending review"
                        if request.submit_for_review
                        else "proposal saved"
                    )
                    context.log(f"{asset.name}: segmentation completed; {saved}.")
                except Cancelled:
                    raise
                except Exception as exc:
                    if not isinstance(exc, DomainError):
                        logger.exception("Batch segmentation failed for %s", asset.id)
                    failures = result["failed"]
                    assert isinstance(failures, list)
                    error = (
                        str(exc)
                        if isinstance(exc, DomainError)
                        else "Segmentation failed; inspect server logs."
                    )
                    failures.append(
                        {
                            "asset_id": asset.id,
                            "name": asset.name,
                            "error": error,
                        }
                    )
                    with self.store.transaction() as session:
                        job = session.get(Job, context.job_id)
                        session.update(job.model_copy(update={"result": result}))
                    context.log(f"{asset.name}: {error}", "error")
                case.progress(1)
            if not result["asset_ids"]:
                raise DomainError(
                    "Segmentation failed for every selected image. See Activity for details."
                )
            completed = result["asset_ids"]
            failed = result["failed"]
            assert isinstance(completed, list) and isinstance(failed, list)
            context.log(f"Finished: {len(completed)} images segmented; {len(failed)} failed.")
            return Outcome(result)

        return self.jobs.submit(
            "batch_annotate",
            project_id,
            request.model_dump(mode="json") | {"asset_ids": asset_ids, "user_id": user_id},
            work,
            guard=guard,
        )
