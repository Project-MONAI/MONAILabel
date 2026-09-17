"""Immutable training inputs, independent evaluation, and scoped model promotion."""

from pydantic import JsonValue

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.evaluation import EvaluationSet, EvaluationSetVersion, ModelSplit
from monailabel.core.models import (
    Asset,
    DeleteModelRequest,
    EvaluateRequest,
    Evaluation,
    Job,
    Learner,
    LearnerCreate,
    LearnerUpdate,
    ModelRecord,
    Project,
    PromoteRequest,
    Promotion,
    Snapshot,
    SnapshotRequest,
    Split,
    StartTraining,
    TrainingMode,
    TrainRequest,
    new_id,
)
from monailabel.core.ports import TrainingProgress, TrainingVolume, Volume, VolumeTrainer
from monailabel.providers.vista3d import mapping as vista_mapping
from monailabel.server.data import Datasets
from monailabel.server.deletion import Deletion
from monailabel.server.evaluation_sets import EvaluationSets, check_training
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.models import Models
from monailabel.server.scoring import ValidationScorer
from monailabel.server.storage import Artifacts, Session, Store
from monailabel.server.training_reports import TrainingReports


class Learning:
    def __init__(
        self,
        store: Store,
        artifacts: Artifacts,
        models: Models,
        jobs: Jobs,
        datasets: Datasets,
        scorer: ValidationScorer,
        reports: TrainingReports,
    ):
        self.store, self.artifacts, self.models, self.jobs = store, artifacts, models, jobs
        self.datasets = datasets
        self.scorer, self.reports = scorer, reports
        self.recipes = models.recipes

    def create(self, project_id: str, request: LearnerCreate) -> Learner:
        project = self.store.get(Project, project_id)
        ids = request.label_ids
        parent = (
            self.models.get(project_id, request.initial_model_id)
            if request.initial_model_id
            else None
        )
        inherited = bool(
            parent
            and parent.provider == request.recipe == "vista3d"
            and (parent.read_only or parent.inherit_targets)
        )
        if request.inherit_targets and not inherited:
            raise DomainError("This initial model does not support inherited targets.")
        if (
            0 not in ids
            or len(set(ids)) != len(ids)
            or not set(ids) <= {label.id for label in project.labels}
        ):
            raise DomainError("Choose project labels, including background once.")
        if len(ids) < 2 and not inherited:
            raise DomainError("Choose at least one foreground target for this training setup.")
        settings = dict(request.config)
        if request.recipe == "monai-unet":
            if request.initial_model_id:
                parent = self.models.get(project_id, request.initial_model_id)
                settings = parent.config | settings
            else:
                layouts = {
                    (len(a.spatial_shape), self.artifacts.array(a.image_key).shape[-1])
                    for a in self.store.list(Asset, project_id)
                }
                if len(layouts) > 1:
                    raise DomainError(
                        "Use a separate project for each image dimension/channel layout."
                    )
                if layouts:
                    dims, channels = layouts.pop()
                    settings.setdefault("spatial_dims", dims)
                    settings.setdefault("in_channels", channels)
        if request.recipe == "vista3d":
            if parent:
                settings = parent.config | settings
            settings["label_mapping"] = {
                str(key): value
                for key, value in vista_mapping(
                    [label for label in project.labels if label.id in ids]
                ).items()
            }
        config = self.recipes.validate(request.recipe, settings)
        if request.initial_model_id:
            parent = self.models.get(project_id, request.initial_model_id)
            if parent.provider != request.recipe:
                raise DomainError("Initial model must use the requested training recipe.")
            if not inherited and parent.label_ids != [0] + [i for i in ids if i]:
                raise DomainError("Initial project model must use the same target mapping.")
        learner = Learner(
            project_id=project_id,
            protocol_version=project.protocol_version,
            name=request.name.strip(),
            recipe=request.recipe,
            label_ids=[0] + [i for i in ids if i],
            config=config,
            initial_model_id=request.initial_model_id,
            inherit_targets=inherited,
        )
        with self.store.transaction() as session:
            for old in session.list(Learner, project_id):
                if old.archived:
                    continue
                if (
                    old.name,
                    old.recipe,
                    old.label_ids,
                    self.recipes.validate(old.recipe, old.config)
                    if old.recipe == learner.recipe
                    else old.config,
                    old.protocol_version,
                    old.initial_model_id,
                    old.inherit_targets,
                ) == (
                    learner.name,
                    learner.recipe,
                    learner.label_ids,
                    learner.config,
                    learner.protocol_version,
                    learner.initial_model_id,
                    learner.inherit_targets,
                ):
                    return old
                if old.name.strip().casefold() == learner.name.casefold():
                    raise DomainError(
                        "Another training setup has this name. Choose a different name."
                    )
            if not learner.name:
                raise DomainError("Give this model a name to use in chat.")
            session.insert(learner)
        return learner

    def update(self, project_id: str, identifier: str, request: LearnerUpdate) -> Learner:
        with self.store.transaction() as session:
            learner = session.get(Learner, identifier)
            if learner.project_id != project_id or learner.archived:
                raise DomainError("Training setup is not available in this project.")
            if learner.version != request.base_version:
                raise Conflict("Training settings changed. Refresh and try again.")
            Deletion.idle(session, project_id)
            name = request.name.strip()
            if not name:
                raise DomainError("Give this model a name to use in chat.")
            if any(
                old.id != identifier
                and not old.archived
                and old.name.strip().casefold() == name.casefold()
                for old in session.list(Learner, project_id)
            ):
                raise DomainError("Another training setup has this name. Choose a different name.")
            updated = learner.model_copy(
                update={
                    "name": name,
                    "version": learner.version + 1,
                }
            )
            session.update(updated)
            # Versions sharing the setup's display name follow its rename. Preserve
            # older versions that were explicitly given a different name.
            for model in session.list(ModelRecord, project_id):
                if (
                    model.learner_id == learner.id
                    and not model.archived
                    and model.name == learner.name
                ):
                    session.update(
                        model.model_copy(update={"name": name, "version": model.version + 1})
                    )
        return updated

    def delete(self, project_id: str, identifier: str, request: DeleteModelRequest) -> Learner:
        with self.store.transaction() as session:
            learner = session.get(Learner, identifier)
            if learner.project_id != project_id or learner.archived:
                raise DomainError("Training setup is not available in this project.")
            if learner.version != request.base_version:
                raise Conflict("Training settings changed. Refresh before deleting.")
            if request.confirmation_name != learner.name:
                raise DomainError("Type the model's exact name to confirm deletion.")
            Deletion.idle(session, project_id)
            if request.scope == "model":
                Deletion.model_family(session, project_id, identifier, request.related_versions)
                return session.get(Learner, identifier)
            # Retain provenance for checkpoints and completed jobs; exclude from new work.
            updated = learner.model_copy(update={"archived": True, "version": learner.version + 1})
            session.update(updated)
        return updated

    def training_config(
        self, learner: Learner, project: Project, label_ids: list[int]
    ) -> dict[str, JsonValue]:
        """Resolve a run's targets without mutating the reusable training setup."""
        if (
            len(label_ids) < 2
            or label_ids[0] != 0
            or len(set(label_ids)) != len(label_ids)
            or not set(label_ids) <= {label.id for label in project.labels}
        ):
            raise DomainError(
                "Choose annotated organs for this training run, including background."
            )
        if learner.recipe != "vista3d":
            if label_ids != learner.label_ids:
                raise DomainError(
                    "This network requires the training setup's fixed target mapping."
                )
            return self.recipes.validate(learner.recipe, learner.config)
        if not learner.inherit_targets and not set(label_ids) <= set(learner.label_ids):
            raise DomainError("Choose targets supported by this training setup.")
        settings: dict[str, JsonValue] = learner.config | {
            "label_mapping": {
                str(key): value
                for key, value in vista_mapping(
                    [label for label in project.labels if label.id in label_ids]
                ).items()
            }
        }
        return self.recipes.validate(learner.recipe, settings)

    def run_config(
        self,
        learner: Learner,
        project: Project,
        label_ids: list[int],
        overrides: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        settings = self.training_config(learner, project, label_ids)
        allowed = {
            "epochs",
            "steps_per_epoch",
            "batch_size",
            "patch_size",
            "learning_rate",
            "weight_decay",
            "device",
            "seed",
            "spacing",
            "intensity_window",
        }
        if set(overrides) - (allowed & settings.keys()):
            raise DomainError(
                "Only this recipe's run settings can be overridden; "
                "architecture and output classes are fixed."
            )
        return self.recipes.validate(learner.recipe, settings | overrides)

    def start(
        self,
        project_id: str,
        learner_id: str,
        request: StartTraining,
        *,
        authorized_by: str | None = None,
    ) -> Job:
        learner = self.store.get(Learner, learner_id)
        project = self.store.get(Project, project_id)
        if (
            learner.archived
            or learner.project_id != project_id
            or learner.protocol_version != project.protocol_version
        ):
            raise DomainError("Training setup belongs to another project or protocol.")
        label_ids = request.label_ids or learner.label_ids
        if (
            request.parent_model_id
            and (request.mode == TrainingMode.CONTINUE or len(label_ids) < 2)
            and request.label_ids is None
        ):
            label_ids = self.models.get(project_id, request.parent_model_id).label_ids
        config = self.run_config(learner, project, label_ids, request.config)
        evaluation_version_id = request.evaluation_version_id
        evaluation_set_id = request.evaluation_set_id
        if (
            not (evaluation_set_id or request.evaluation_version_id or request.snapshot_id)
            and request.validation_percentage is None
        ):
            evaluation_set_id = learner.evaluation_set_id
        saved_split = next(
            (
                item
                for item in self.store.list(ModelSplit, project_id)
                if item.learner_id == learner.id
            ),
            None,
        )
        # Preserve explicit legacy snapshots/references; shared data and existing
        # model-owned memberships always use this model's growing split.
        model_split = request.validation_percentage is not None or (
            not (evaluation_set_id or evaluation_version_id or request.snapshot_id)
            and (
                saved_split is not None
                or not any(a.split == Split.TRAIN for a in self.store.list(Asset, project_id))
            )
        )
        if (
            not evaluation_set_id
            and not evaluation_version_id
            and not request.snapshot_id
            and not model_split
        ):
            active = [
                item for item in self.store.list(EvaluationSet, project_id) if not item.archived
            ]
            if len(active) > 1:
                raise DomainError("Choose an evaluation set for this training run.")
            if active:
                evaluation_set_id = active[0].id
        if evaluation_set_id:
            evaluation_version_id = (
                EvaluationSets(self.store)
                .for_training(project_id, evaluation_set_id, label_ids, authorized_by or "system")
                .id
            )
        reference = (
            self.store.get(EvaluationSetVersion, evaluation_version_id)
            if evaluation_version_id
            else None
        )
        if reference and (
            reference.project_id != project_id
            or reference.protocol_version != project.protocol_version
            or not set(label_ids) <= {label.id for label in reference.labels}
        ):
            raise DomainError(
                "The evaluation version must belong to this project and cover the training "
                "structures."
            )
        snapshot = (
            self.store.get(Snapshot, request.snapshot_id)
            if request.snapshot_id
            else self.datasets.snapshot(
                project_id,
                request=SnapshotRequest(
                    sample_filter=request.sample_filter,
                    label_ids=label_ids,
                    allow_unreviewed_training=request.allow_unreviewed_training,
                    note=request.note,
                ),
                authorized_by=authorized_by,
                learner_id=learner.id if model_split or reference else None,
                external_validation=reference is not None,
                validation_percentage=request.validation_percentage
                or (saved_split.validation_percentage if saved_split else 20),
                parent_model_id=request.parent_model_id,
            )
        )
        if reference:
            if snapshot.project_id != project_id:
                raise DomainError("Snapshot belongs to another project.")
            snapshot = snapshot.model_copy(
                update={
                    "id": new_id(),
                    "evaluation_version_id": reference.id,
                    "samples": [s for s in snapshot.samples if s.split == Split.TRAIN]
                    + reference.samples,
                }
            )
            with self.store.transaction() as session:
                session.insert(snapshot)
        if not any(s.split == Split.VALIDATION for s in snapshot.samples):
            raise DomainError(
                "Accept at least one complete held-out validation case before starting this model."
            )
        job = self.train(
            project_id,
            TrainRequest(
                snapshot_id=snapshot.id,
                recipe=learner.recipe,
                config=config,
                label_ids=label_ids,
                learner_id=learner.id,
                name=learner.name,
                mode=request.mode,
                parent_model_id=request.parent_model_id,
            ),
        )

        with self.store.transaction() as session:
            current = session.get(Learner, learner.id)
            if current.evaluation_set_id != evaluation_set_id:
                session.update(
                    current.model_copy(
                        update={
                            "evaluation_set_id": evaluation_set_id,
                            "version": current.version + 1,
                        }
                    )
                )
        return job

    def train(self, project_id: str, request: TrainRequest, key: str | None = None) -> Job:
        project = self.store.get(Project, project_id)
        config = self.recipes.validate(request.recipe, request.config)
        snapshot = self.store.get(Snapshot, request.snapshot_id)
        if (
            snapshot.project_id != project.id
            or snapshot.protocol_version != project.protocol_version
        ):
            raise DomainError("Snapshot does not match this project's protocol.")
        if snapshot.model_split_id and snapshot.model_split_id != request.learner_id:
            raise DomainError("This snapshot belongs to another model's training split.")
        samples = [s for s in snapshot.samples if s.split == Split.TRAIN]
        if any(s.label_source == "model_prediction" for s in samples) and (
            not snapshot.allow_unreviewed_training or not snapshot.authorized_by
        ):
            raise DomainError(
                "Training on model predictions requires a manager-authorized snapshot."
            )
        label_ids = request.label_ids or [label.id for label in snapshot.labels]
        if (
            0 not in label_ids
            or len(label_ids) < 2
            or len(set(label_ids)) != len(label_ids)
            or not set(label_ids) <= {label.id for label in snapshot.labels}
        ):
            raise DomainError(
                "Training labels must be unique labels from the snapshot, with background."
            )
        learner = None
        if request.learner_id:
            learner = self.store.get(Learner, request.learner_id)
            if (
                learner.archived
                or learner.project_id != project_id
                or learner.protocol_version != project.protocol_version
                or learner.recipe != request.recipe
            ):
                raise DomainError("Training request does not match its project model setup.")
            recommended = self.training_config(learner, project, label_ids)
            overrides = {k: v for k, v in config.items() if v != recommended.get(k)}
            if self.run_config(learner, project, label_ids, overrides) != config:
                raise DomainError("Training request does not match its project model setup.")
        parent = None
        if request.parent_model_id:
            parent = self.models.get(project.id, request.parent_model_id)
            base_vista = parent.provider == "vista3d" and parent.read_only
            inherited_vista = (
                parent.provider == "vista3d"
                and parent.inherit_targets
                and request.mode == TrainingMode.FINE_TUNE
            )
            if parent.provider != request.recipe or (
                not (base_vista or inherited_vista) and parent.label_ids != label_ids
            ):
                raise DomainError("Parent must use this recipe and the same ordered label mapping.")
            if base_vista and request.mode != TrainingMode.FINE_TUNE:
                raise DomainError("Fine-tune the read-only VISTA3D base into a new project model.")
            for sample in samples:
                if (
                    request.recipe == "pixel-gaussian"
                    and sample.asset_id in parent.training_revisions
                    and (sample.revision != parent.training_revisions[sample.asset_id])
                ):
                    raise DomainError("Previously trained labels changed; retrain from scratch.")
            if request.recipe == "pixel-gaussian":
                samples = [s for s in samples if s.asset_id not in parent.training_assets]
        if not samples:
            raise DomainError(
                "No new eligible training samples. Pool and validation data are excluded."
            )

        def work(context: JobContext) -> Outcome:
            context.log(
                f"{request.name} · {request.recipe} · {request.mode.value} · "
                f"{len(samples)} training cases."
            )
            filters = snapshot.sample_filter
            if filters.source_ids or filters.asset_ids or filters.limit:
                context.log(
                    "Training filters · "
                    + (f"{len(filters.source_ids)} sources · " if filters.source_ids else "")
                    + (f"{len(filters.asset_ids)} selected images · " if filters.asset_ids else "")
                    + (f"limit {filters.limit} images · " if filters.limit else "")
                    + f"{len(samples)} eligible training images; evaluation unchanged."
                )
            if config:
                context.log(
                    f"{config.get('epochs')} epochs × {config.get('steps_per_epoch')} steps · "
                    f"batch size {config.get('batch_size', 1)} · "
                    f"learning rate {config.get('learning_rate')}."
                )
            progress = TrainingProgress(lambda value: context.progress(0.8 * value), context.log)
            context.log("Preparing training data and model weights.")
            state = self.artifacts.json(parent.state_key) if parent and parent.state_key else None
            if parent and parent.provider == "vista3d" and parent.read_only:
                state = {"format": "vista3d-base-v1"}
            trainer = self.recipes.trainer(request.recipe, config)
            if isinstance(trainer, VolumeTrainer):
                volumes = []
                for sample in samples:
                    affine = sample.affine or self.store.get(Asset, sample.asset_id).affine
                    if affine is None:
                        raise DomainError("This training recipe requires source volume geometry.")
                    volumes.append(
                        TrainingVolume(
                            Volume(self.artifacts.array(sample.image_key), affine),
                            self.artifacts.array(sample.mask_key),
                        )
                    )
                state = trainer.train_volumes(volumes, label_ids, request.mode, state, progress)
            else:
                examples = (
                    (self.artifacts.array(s.image_key), self.artifacts.array(s.mask_key))
                    for s in samples
                )
                state = trainer.train(examples, label_ids, request.mode, state, progress)
            model = ModelRecord(
                project_id=project.id,
                name=request.name,
                provider=request.recipe,
                label_ids=label_ids,
                state_key=self.artifacts.put_json(state),
                config=config,
                learner_id=request.learner_id,
                parent_id=parent.id if parent else None,
                snapshot_id=snapshot.id,
                training_assets=sorted(
                    set(parent.training_assets if parent else []) | {s.asset_id for s in samples}
                ),
                training_groups=sorted(
                    set(parent.training_groups if parent else []) | {s.group_id for s in samples}
                ),
                training_revisions=(parent.training_revisions if parent else {})
                | {s.asset_id: s.revision for s in samples},
                mode=request.mode,
                inherit_targets=bool(
                    request.recipe == "vista3d"
                    and parent
                    and (parent.read_only or parent.inherit_targets)
                ),
                unreviewed_training=bool(parent and parent.unreviewed_training)
                or any(s.label_source == "model_prediction" for s in samples),
            )
            context.log("Training finished. Evaluating the saved checkpoint on held-out cases.")
            report = self.reports.build(
                project, model, snapshot, context.job_id, context, start=0.8
            )
            return Outcome(
                {
                    "training_report_id": report.id,
                    "model_id": model.id,
                    "snapshot_id": snapshot.id,
                    "training_samples": len(samples),
                },
                [model, report],
            )

        def guard(session: Session) -> None:
            if request.learner_id:
                current_learner = session.get(Learner, request.learner_id)
                if current_learner.archived:
                    raise Conflict("This model was deleted before training could start.")
                if any(
                    job.kind == "train"
                    and job.status in {"queued", "running"}
                    and job.request.get("learner_id") == request.learner_id
                    for job in session.list(Job, project.id)
                ):
                    raise Conflict("This model is already training. Wait for its current run.")
            check_training(session, project.id, samples, request.learner_id)
            if snapshot.model_split_id:
                current = session.get(ModelSplit, snapshot.model_split_id)
                if current.version != snapshot.model_split_version:
                    raise Conflict("This model's split changed. Refresh before training.")

        return self.jobs.submit(
            "train",
            project.id,
            request.model_dump(mode="json"),
            work,
            key,
            guard=guard,
        )

    def evaluate(
        self,
        project_id: str,
        request: EvaluateRequest,
        key: str | None = None,
        *,
        authorized_by: str | None = None,
    ) -> Job:
        project = self.store.get(Project, project_id)
        candidate = self.models.get(project_id, request.candidate_id)
        baseline = self.models.get(project_id, request.baseline_id)
        shared = self.models.supported_labels(project, candidate) & self.models.supported_labels(
            project, baseline
        )
        # Prefer the candidate's trained targets. A base model's stored [0] is
        # not its vocabulary; resolve supported project structures separately.
        targets = set(candidate.label_ids) - {0} or set(baseline.label_ids) - {0} or shared
        label_ids = request.label_ids or sorted(targets & shared)
        if not label_ids or len(set(label_ids)) != len(label_ids) or not set(label_ids) <= shared:
            raise DomainError("Choose models with shared foreground structures to evaluate.")
        candidate = self.models.for_labels(candidate, label_ids)
        baseline = self.models.for_labels(baseline, label_ids)
        set_id = request.evaluation_set_id
        if not set_id and not request.evaluation_version_id and not request.snapshot_id:
            active = [s for s in self.store.list(EvaluationSet, project_id) if not s.archived]
            if len(active) > 1:
                raise DomainError("Choose an evaluation set for this comparison.")
            set_id = active[0].id if active else None
        if set_id:
            snapshot: Snapshot | EvaluationSetVersion = EvaluationSets(self.store).for_training(
                project_id, set_id, [0, *label_ids], authorized_by or "system"
            )
        elif request.evaluation_version_id:
            snapshot = self.store.get(EvaluationSetVersion, request.evaluation_version_id)
        elif request.snapshot_id:
            snapshot = self.store.get(Snapshot, request.snapshot_id)
        else:
            snapshot = self.datasets.snapshot(
                project_id, request=SnapshotRequest(label_ids=[0, *label_ids])
            )
        validation = self.scorer.samples(project, candidate, snapshot, label_ids)
        self.scorer.samples(project, baseline, snapshot, label_ids)

        def work(context: JobContext) -> Outcome:
            names = ", ".join(label.name for label in project.labels if label.id in label_ids)
            context.log(
                f"Comparing {candidate.name} with {baseline.name} on {len(validation)} "
                f"held-out cases. Structures: {names}."
            )
            scores = []
            for index, model in enumerate((candidate, baseline)):
                context.log(f"Evaluating model {index + 1} of 2: {model.name}.")

                def progress(
                    sample_index: int,
                    count: int,
                    model_index: int = index,
                    model_name: str = model.name,
                ) -> None:
                    context.progress((model_index + sample_index / count) / 2)
                    context.log(
                        f"{model_name}: evaluating held-out case {sample_index + 1} of {count}."
                    )

                score, _ = self.scorer.score(
                    project,
                    model,
                    validation,
                    label_ids,
                    progress,
                )
                scores.append(score)
                context.log(f"{model.name}: mean Dice {score.mean_dice:.4f}.")
                for label in project.labels:
                    if label.id in label_ids:
                        context.log(
                            f"{model.name} · {label.name}: Dice {score.per_class[label.id]:.4f}."
                        )
            candidate_score, baseline_score = scores
            eligible = [
                label
                for label in label_ids
                if (
                    candidate_score.per_class[label] >= request.min_dice
                    and candidate_score.per_class[label] - baseline_score.per_class[label]
                    >= request.min_improvement
                )
            ]
            evaluation = Evaluation(
                project_id=project.id,
                snapshot_id=snapshot.id if isinstance(snapshot, Snapshot) else None,
                evaluation_version_id=snapshot.id
                if isinstance(snapshot, EvaluationSetVersion)
                else snapshot.evaluation_version_id,
                candidate_id=candidate.id,
                baseline_id=baseline.id,
                candidate=candidate_score,
                baseline=baseline_score,
                eligible_labels=eligible,
                min_dice=request.min_dice,
                min_improvement=request.min_improvement,
                validation_assets=[s.asset_id for s in validation],
            )
            return Outcome({"evaluation_id": evaluation.id}, [evaluation])

        return self.jobs.submit("evaluate", project.id, request.model_dump(mode="json"), work, key)

    def promote(self, project_id: str, request: PromoteRequest) -> Promotion:
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            evaluation = session.get(Evaluation, request.evaluation_id)
            if evaluation.project_id != project_id:
                raise DomainError("Evaluation belongs to another project.")
            candidate = session.get(ModelRecord, evaluation.candidate_id)
            if candidate.archived:
                raise DomainError("Deleted models cannot be promoted.")
            if project.version != request.base_version:
                raise Conflict("Project defaults changed. Reload before promoting.")
            if len(set(request.label_ids)) != len(request.label_ids) or not set(
                request.label_ids
            ) <= set(evaluation.eligible_labels):
                raise DomainError(
                    "Selected classes did not pass the evaluation promotion criteria."
                )
            if any(
                project.defaults.get(label) != evaluation.baseline_id for label in request.label_ids
            ):
                raise Conflict("Evaluate against the currently selected baseline before promotion.")
            promotion = Promotion(
                project_id=project_id,
                evaluation_id=evaluation.id,
                model_id=evaluation.candidate_id,
                label_ids=request.label_ids,
                previous_defaults=project.defaults,
                project_version=project.version + 1,
            )
            session.insert(promotion)
            session.update(
                project.model_copy(
                    update={
                        "defaults": project.defaults
                        | {label: evaluation.candidate_id for label in request.label_ids},
                        "version": promotion.project_version,
                    }
                )
            )
            return promotion

    def rollback(self, promotion_id: str) -> Project:
        with self.store.transaction() as session:
            promotion = session.get(Promotion, promotion_id)
            project = session.get(Project, promotion.project_id)
            if promotion.reverted or project.version != promotion.project_version:
                raise Conflict("Only the current, unreverted promotion can be rolled back.")
            if any(
                session.get(ModelRecord, model_id).archived
                for model_id in promotion.previous_defaults.values()
            ):
                raise DomainError(
                    "A previous default was deleted. Choose an available default instead."
                )
            project = project.model_copy(
                update={
                    "defaults": promotion.previous_defaults,
                    "version": project.version + 1,
                }
            )
            session.update(project)
            session.update(promotion.model_copy(update={"reverted": True}))
            return project
