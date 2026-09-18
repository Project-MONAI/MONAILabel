"""Saved evaluation membership and references, independent of model training runs."""

import hashlib
import math

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.evaluation import (
    EvaluationReservation,
    EvaluationSet,
    EvaluationSetCreate,
    EvaluationSetDelete,
    EvaluationSetExtend,
    EvaluationSetPublish,
    EvaluationSetUpdate,
    EvaluationSetVersion,
    ModelSplit,
)
from monailabel.core.models import (
    Annotation,
    Asset,
    Job,
    ModelRecord,
    Project,
    ReviewDecision,
    Sample,
    Snapshot,
    Split,
)
from monailabel.core.video import VideoAsset
from monailabel.server.storage import Session, Store


def components(assets: list[Asset]) -> dict[str, list[Asset]]:
    """Keep declared patient groups and exact decoded-image duplicates together."""
    parents = {a.group_id: a.group_id for a in assets}

    def root(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    images: dict[str, str] = {}
    for asset in assets:
        other = images.setdefault(asset.image_key, asset.group_id)
        a, b = root(asset.group_id), root(other)
        parents[max(a, b)] = min(a, b)
    result: dict[str, list[Asset]] = {}
    for asset in assets:
        result.setdefault(root(asset.group_id), []).append(asset)
    return result


def reserved(session: Session, project_id: str) -> tuple[set[str], set[str]]:
    records = session.list(EvaluationReservation, project_id)
    groups = {r.group_id for r in records}
    groups.update(
        v.group_id for v in session.list(VideoAsset, project_id) if v.split == Split.VALIDATION
    )
    return groups, {key for r in records for key in r.image_keys}


def check_training(
    session: Session, project_id: str, samples: list[Sample], learner_id: str | None = None
) -> None:
    groups, images = reserved(session, project_id)
    if learner_id:
        for record in session.list(ModelSplit, project_id):
            if record.learner_id == learner_id:
                groups.update(record.validation_groups)
                images.update(record.validation_image_keys)
    # Also guard ordinary evaluation assignments, including old snapshots.
    for asset in session.list(Asset, project_id):
        if asset.split == Split.VALIDATION:
            groups.add(asset.group_id)
            images.add(asset.image_key)
    if any(s.group_id in groups or s.image_key in images for s in samples):
        raise Conflict(
            "Evaluation-only cases cannot be used for training, including older snapshots."
        )


def training_history(session: Session, project_id: str) -> tuple[set[str], set[str]]:
    groups = {g for m in session.list(ModelRecord, project_id) for g in m.training_groups}
    snapshots = {s.id: s for s in session.list(Snapshot, project_id)}
    images: set[str] = set()
    # Includes queued/running jobs: reservation and job creation share a transaction lock.
    for job in session.list(Job, project_id):
        identifier = job.request.get("snapshot_id")
        if job.kind == "train" and isinstance(identifier, str) and identifier in snapshots:
            for sample in snapshots[identifier].samples:
                if sample.split == Split.TRAIN:
                    groups.add(sample.group_id)
                    images.add(sample.image_key)
    for asset in session.list(Asset, project_id):
        if asset.group_id in groups:
            images.add(asset.image_key)
    return groups, images


class EvaluationSets:
    def __init__(self, store: Store):
        self.store = store

    @staticmethod
    def get(session: Session, project_id: str, identifier: str) -> EvaluationSet:
        record = session.get(EvaluationSet, identifier)
        if record.project_id != project_id:
            raise DomainError("Evaluation set belongs to another project.")
        return record

    @staticmethod
    def check_version(record: EvaluationSet, version: int) -> None:
        if record.version != version:
            raise Conflict("This evaluation set changed. Refresh and try again.")

    @staticmethod
    def check_name(session: Session, project_id: str, name: str, identifier: str = "") -> str:
        name = name.strip()
        if not name:
            raise DomainError("Give the evaluation set a name.")
        if any(
            s.id != identifier and s.name.casefold() == name.casefold()
            for s in session.list(EvaluationSet, project_id)
        ):
            raise Conflict("An evaluation set already has this name, including archived sets.")
        return name

    @staticmethod
    def choose_assets(session: Session, project_id: str, ids: list[str] | None) -> list[Asset]:
        assets = session.list(Asset, project_id)
        if ids is not None and not set(ids) <= {a.id for a in assets}:
            raise DomainError("Choose samples from this project.")
        return assets if ids is None else [a for a in assets if a.id in ids]

    @staticmethod
    def balance(
        session: Session,
        record: EvaluationSet,
        additions: list[Asset],
        *,
        include_all: bool = False,
    ) -> EvaluationSet:
        all_assets = session.list(Asset, record.project_id)
        grouped = components(all_assets)
        cohort = set(record.cohort_groups) | {a.group_id for a in additions}
        members = set(record.member_groups)
        selected = {
            key for key, values in grouped.items() if any(a.group_id in members for a in values)
        }
        eligible_groups = {
            key: values
            for key, values in grouped.items()
            if any(a.group_id in cohort for a in values)
        }
        used_groups, used_images = training_history(session, record.project_id)
        candidates = [
            key
            for key, values in eligible_groups.items()
            if key not in selected
            and not any(a.group_id in used_groups or a.image_key in used_images for a in values)
        ]
        candidates.sort(key=lambda key: hashlib.sha256(f"{record.seed}:{key}".encode()).digest())
        if include_all:
            addition_ids = {item.id for item in additions}
            requested = {
                key
                for key, values in eligible_groups.items()
                if any(a.id in addition_ids for a in values)
            }
            if any(
                a.group_id in used_groups or a.image_key in used_images
                for key in requested
                for a in eligible_groups[key]
            ):
                raise Conflict(
                    "Some selected cases were used for training and cannot be evaluation-only."
                )
            selected.update(requested)
        else:
            target = math.ceil(len(eligible_groups) * record.percentage / 100)
            selected.update(candidates[: max(0, target - len(selected))])
        reservations = {
            r.group_id: r for r in session.list(EvaluationReservation, record.project_id)
        }
        reserved_groups, reserved_images = reserved(session, record.project_id)
        videos = session.list(VideoAsset, record.project_id)
        for key, values in eligible_groups.items():
            for asset in values:
                cohort.add(asset.group_id)
                if key in selected:
                    members.add(asset.group_id)
                    old = reservations.get(asset.group_id)
                    keys = sorted(
                        set(old.image_keys if old else []) | {a.image_key for a in values}
                    )
                    if old and keys != old.image_keys:
                        updated = old.model_copy(update={"image_keys": keys})
                        session.update(updated)
                        reservations[asset.group_id] = updated
                    elif not old:
                        reservation = EvaluationReservation(
                            project_id=record.project_id, group_id=asset.group_id, image_keys=keys
                        )
                        session.insert(reservation)
                        reservations[asset.group_id] = reservation
                    split = Split.VALIDATION
                elif asset.group_id in reserved_groups or asset.image_key in reserved_images:
                    split = Split.VALIDATION
                else:
                    split = (
                        Split.TRAIN
                        if asset.split == Split.POOL and not include_all
                        else asset.split
                    )
                if split != asset.split:
                    session.update(asset.model_copy(update={"split": split}))
                for video in videos:
                    if video.group_id == asset.group_id and video.split != split:
                        session.update(video.model_copy(update={"split": split}))
        updates = {"cohort_groups": sorted(cohort), "member_groups": sorted(members)}
        if updates != {key: getattr(record, key) for key in updates}:
            record = record.model_copy(update=updates | {"version": record.version + 1})
        return record

    @classmethod
    def on_import(
        cls, session: Session, asset: Asset, *, is_new: bool, apply_policies: bool = True
    ) -> Asset:
        """Called inside image import's transaction; no sample can escape its reservation."""
        groups, images = reserved(session, asset.project_id)
        if asset.group_id in groups or asset.image_key in images:
            if asset.split != Split.VALIDATION:
                asset = asset.model_copy(update={"split": Split.VALIDATION})
                session.update(asset)
            # Preserve the alias even if the original set was deleted or archived.
            existing = next(
                (
                    r
                    for r in session.list(EvaluationReservation, asset.project_id)
                    if r.group_id == asset.group_id
                ),
                None,
            )
            keys = sorted(set(existing.image_keys if existing else []) | {asset.image_key})
            if existing:
                session.update(existing.model_copy(update={"image_keys": keys}))
            else:
                session.insert(
                    EvaluationReservation(
                        project_id=asset.project_id, group_id=asset.group_id, image_keys=keys
                    )
                )
        if is_new and apply_policies:
            for record in session.list(EvaluationSet, asset.project_id):
                if record.auto_update and not record.archived:
                    session.update(cls.balance(session, record, [asset]))
        return session.get(Asset, asset.id)

    def create(self, project_id: str, request: EvaluationSetCreate) -> EvaluationSet:
        with self.store.transaction() as session:
            session.get(Project, project_id)
            record = EvaluationSet(
                project_id=project_id,
                name=self.check_name(session, project_id, request.name),
                percentage=request.percentage,
                auto_update=request.auto_update,
            )
            record = self.balance(
                session,
                record,
                self.choose_assets(session, project_id, request.asset_ids),
            )
            session.insert(record)
            return record

    def update(
        self, project_id: str, identifier: str, request: EvaluationSetUpdate
    ) -> EvaluationSet:
        with self.store.transaction() as session:
            record = self.get(session, project_id, identifier)
            self.check_version(record, request.base_version)
            updates = request.model_dump(exclude={"base_version"}, exclude_none=True)
            if "name" in updates:
                updates["name"] = self.check_name(
                    session, project_id, str(updates["name"]), identifier
                )
            if not updates or all(getattr(record, key) == value for key, value in updates.items()):
                return record
            record = record.model_copy(update=updates | {"version": record.version + 1})
            if not record.archived and request.percentage is not None:
                record = self.balance(session, record, [])
            session.update(record)
            return record

    def extend(
        self, project_id: str, identifier: str, request: EvaluationSetExtend
    ) -> EvaluationSet:
        with self.store.transaction() as session:
            record = self.get(session, project_id, identifier)
            self.check_version(record, request.base_version)
            if record.archived:
                raise Conflict("Restore this evaluation set before adding cases.")
            record = self.balance(
                session,
                record,
                self.choose_assets(session, project_id, request.asset_ids),
                include_all=request.include_all,
            )
            session.update(record)
            return record

    def delete(self, project_id: str, identifier: str, request: EvaluationSetDelete) -> None:
        with self.store.transaction() as session:
            record = self.get(session, project_id, identifier)
            self.check_version(record, request.base_version)
            if any(
                v.evaluation_set_id == identifier
                for v in session.list(EvaluationSetVersion, project_id)
            ):
                raise Conflict(
                    "This set has saved reference versions. Archive it to keep learning history."
                )
            session.connection.execute(
                "DELETE FROM records WHERE kind='EvaluationSet' AND id=?", (identifier,)
            )
            # Reservations intentionally survive deletion. Deleting a name never releases eval data.

    def for_training(
        self, project_id: str, identifier: str, label_ids: list[int], user_id: str
    ) -> EvaluationSetVersion:
        """Reuse fixed references, preparing the first compatible version when needed."""
        with self.store.transaction() as session:
            record = self.get(session, project_id, identifier)
            if record.archived:
                raise Conflict("Restore this evaluation set before using it for training.")
            project = session.get(Project, project_id)
            if record.latest_version_id:
                version = session.get(EvaluationSetVersion, record.latest_version_id)
                if version.protocol_version == project.protocol_version and set(label_ids) <= {
                    label.id for label in version.labels
                }:
                    return version
        # Publication checks the observed set revision and all accepted annotations
        # transactionally. A concurrent membership edit must not be silently applied.
        return self.publish(
            project_id,
            identifier,
            EvaluationSetPublish(base_version=record.version, label_ids=label_ids),
            user_id,
        )

    def publish(
        self, project_id: str, identifier: str, request: EvaluationSetPublish, user_id: str
    ) -> EvaluationSetVersion:
        with self.store.transaction() as session:
            record = self.get(session, project_id, identifier)
            self.check_version(record, request.base_version)
            if record.archived:
                raise Conflict("Restore this evaluation set before saving a version.")
            project = session.get(Project, project_id)
            required = set(request.label_ids)
            if (
                0 not in required
                or len(required) < 2
                or len(required) != len(request.label_ids)
                or not required <= {label.id for label in project.labels}
            ):
                raise DomainError("Choose project structures with background for evaluation.")
            assets = [
                a for a in session.list(Asset, project_id) if a.group_id in record.member_groups
            ]
            if not assets:
                raise DomainError("This evaluation set has no reserved cases yet.")
            decisions = {d.annotation_id: d for d in session.list(ReviewDecision, project_id)}
            samples = []
            for asset in assets:
                if not asset.annotation_id:
                    raise DomainError(
                        f"Review a complete reference annotation for {asset.name} first."
                    )
                annotation = session.get(Annotation, asset.annotation_id)
                decision = decisions.get(annotation.id)
                if (
                    decision is None
                    or decision.verdict != "accepted"
                    or not required <= set(annotation.covered_labels)
                ):
                    raise DomainError(f"Accept complete reference labels for {asset.name} first.")
                samples.append(
                    Sample(
                        asset_id=asset.id,
                        image_key=asset.image_key,
                        mask_key=annotation.mask_key,
                        revision=annotation.revision,
                        group_id=asset.group_id,
                        split=Split.VALIDATION,
                        decision_id=decision.id,
                        affine=asset.affine,
                    )
                )
            samples.sort(key=lambda s: s.asset_id)
            labels = [label for label in project.labels if label.id in required]
            versions = [
                v
                for v in session.list(EvaluationSetVersion, project_id)
                if v.evaluation_set_id == identifier
            ]
            for old in reversed(versions):
                if (
                    old.samples == samples
                    and old.labels == labels
                    and old.protocol_version == project.protocol_version
                ):
                    return old
            version = EvaluationSetVersion(
                project_id=project_id,
                evaluation_set_id=identifier,
                number=len(versions) + 1,
                protocol_version=project.protocol_version,
                labels=labels,
                samples=samples,
                published_by=user_id,
            )
            session.insert(version)
            session.update(
                record.model_copy(
                    update={"latest_version_id": version.id, "version": record.version + 1}
                )
            )
            return version
