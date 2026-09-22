"""A training result report, also available for checkpoints created before reporting."""

import logging

from monailabel.core.errors import Cancelled, DomainError
from monailabel.core.models import Job, ModelRecord, Project, Snapshot, TrainingReport
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.scoring import ValidationScorer
from monailabel.server.storage import Artifacts, Store

logger = logging.getLogger(__name__)


class TrainingReports:
    def __init__(self, store: Store, artifacts: Artifacts, scorer: ValidationScorer, jobs: Jobs):
        self.store, self.artifacts, self.scorer, self.jobs = store, artifacts, scorer, jobs

    def build(
        self,
        project: Project,
        model: ModelRecord,
        snapshot: Snapshot,
        training_job_id: str,
        context: JobContext,
        start: float = 0,
    ) -> TrainingReport:
        labels = [label for label in snapshot.labels if label.id in model.label_ids and label.id]
        state = self.artifacts.json(model.state_key) if model.state_key else {}
        initial, final = state.get("initial_loss"), state.get("final_loss")
        report = TrainingReport(
            evaluation_requested=snapshot.evaluation_requested,
            model_split_id=snapshot.model_split_id,
            model_split_version=snapshot.model_split_version,
            project_id=project.id,
            training_job_id=training_job_id,
            model_id=model.id,
            snapshot_id=snapshot.id,
            evaluation_version_id=snapshot.evaluation_version_id,
            labels=labels,
            initial_loss=float(initial) if isinstance(initial, (int, float)) else None,
            final_loss=float(final) if isinstance(final, (int, float)) else None,
        )
        if not snapshot.evaluation_requested:
            return report
        try:
            ids = [label.id for label in labels]
            validation = self.scorer.samples(project, model, snapshot, ids)
            report = report.model_copy(
                update={"validation_assets": sorted({s.asset_id for s in validation})}
            )

            def progress(index: int, count: int) -> None:
                context.progress(start + (0.99 - start) * index / count)
                context.log(f"Evaluating held-out case {index + 1} of {count}.")

            metrics, iou = self.scorer.score(
                project, model, validation, ids, progress, context.check_cancelled
            )
            context.log(f"Evaluation complete. Mean Dice: {metrics.mean_dice:.4f}.")
            return report.model_copy(update={"metrics": metrics, "per_class_iou": iou})
        except Cancelled:
            raise
        except Exception as error:
            if not isinstance(error, DomainError):
                logger.exception("Training report failed for model %s", model.id)
            message = (
                str(error)
                if isinstance(error, DomainError)
                else "Evaluation failed; inspect server logs and retry the report."
            )
            context.log(f"Model saved; evaluation unavailable: {message}")
            return report.model_copy(update={"error": message})

    def get(self, training_job_id: str) -> TrainingReport | None:
        job = self.store.get(Job, training_job_id)
        if job.kind == "training_report":
            identifier = job.result.get("training_report_id")
            return (
                self.store.get(TrainingReport, identifier) if isinstance(identifier, str) else None
            )
        return next(
            (
                r
                for r in reversed(self.store.list(TrainingReport, job.project_id))
                if r.training_job_id == job.id
            ),
            None,
        )

    def start(self, training_job_id: str) -> Job:
        training = self.store.get(Job, training_job_id)
        if training.kind != "train" or training.status != "succeeded":
            raise DomainError("Choose a completed training run.")
        model_id = training.result.get("model_id")
        if not isinstance(model_id, str):
            raise DomainError("This run has no saved model to evaluate.")
        model = self.store.get(ModelRecord, model_id)
        if model.archived or model.project_id != training.project_id or not model.snapshot_id:
            raise DomainError("The trained model or its evaluation references are unavailable.")
        project = self.store.get(Project, training.project_id)
        snapshot = self.store.get(Snapshot, model.snapshot_id)
        # Missing/unreviewed/overlapping references are rejected before a job is queued.
        self.scorer.samples(project, model, snapshot, [label for label in model.label_ids if label])
        prior = self.get(training_job_id)

        def work(context: JobContext) -> Outcome:
            report = self.build(project, model, snapshot, training_job_id, context)
            return Outcome(
                {"training_report_id": report.id, "training_job_id": training_job_id}, [report]
            )

        return self.jobs.submit(
            "training_report",
            project.id,
            {"training_job_id": training_job_id},
            work,
            key=f"training-report:{training_job_id}:{prior.id if prior else 'first'}",
        )
