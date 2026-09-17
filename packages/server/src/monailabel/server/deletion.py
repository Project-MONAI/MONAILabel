"""Atomic workspace deletion; shared artifact cleanup runs only before workers start."""

import json
import logging
import re
from typing import Any

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.evaluation import EvaluationReservation, EvaluationSet, EvaluationSetVersion
from monailabel.core.models import (
    TERMINAL_STATUSES,
    Asset,
    DeletionResult,
    Job,
    Learner,
    ModelRecord,
    Project,
    Snapshot,
)
from monailabel.server.storage import Artifacts, Session, Store, referenced_assets

logger = logging.getLogger(__name__)
ARTIFACT_KEY = re.compile(r"[0-9a-f]{64}\.(npy|json|bin)")


class Deletion:
    def __init__(self, store: Store):
        self.store = store

    @staticmethod
    def idle(session: Session, project_id: str) -> None:
        if any(job.status not in TERMINAL_STATUSES for job in session.list(Job, project_id)):
            raise Conflict("Finish or cancel this project's active jobs before deleting data.")

    @staticmethod
    def model_family(
        session: Session,
        project_id: str,
        learner_id: str,
        related_versions: dict[str, int] | None,
    ) -> None:
        """Remove one named project model from both catalogs, retaining its history."""
        learner = session.get(Learner, learner_id)
        if learner.project_id != project_id:
            raise DomainError("Training setup belongs to another project.")
        models = [
            m
            for m in session.list(ModelRecord, project_id)
            if m.learner_id == learner.id and not m.archived
        ]
        records: list[ModelRecord | Learner] = list(models)
        if not learner.archived:
            records.append(learner)
        if related_versions is not None and related_versions != {
            record.id: record.version for record in records
        }:
            raise Conflict("This model or its trained versions changed. Refresh before deleting.")
        if any(model.preset for model in models):
            raise DomainError("Preloaded base models cannot be deleted.")
        ids = {model.id for model in models}
        project = session.get(Project, project_id)
        if project.annotation_model_id in ids or ids.intersection(project.defaults.values()):
            raise DomainError(
                "A trained version is an annotation default. Choose another default before "
                "deleting this model."
            )
        if any(
            other.id != learner.id and not other.archived and other.initial_model_id in ids
            for other in session.list(Learner, project_id)
        ):
            raise DomainError(
                "Another training setup uses a version of this model as its starting model. "
                "Delete that setup first."
            )
        for record in records:
            session.update(
                record.model_copy(update={"archived": True, "version": record.version + 1})
            )

    @staticmethod
    def rows(session: Session, project_id: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [
            (kind, identifier, json.loads(data))
            for kind, identifier, data in session.connection.execute(
                "SELECT kind,id,data FROM records WHERE project_id=?", (project_id,)
            )
        ]

    @staticmethod
    def remove(session: Session, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        for kind, identifier, _ in rows:
            if kind == "Credential":
                session.connection.execute(
                    "DELETE FROM records WHERE kind='EncryptedCredential' AND id=?", (identifier,)
                )
            if kind == "Conversation":
                # Older receipts did not carry project_id or asset_id.
                session.connection.execute(
                    "DELETE FROM records WHERE kind='ToolExecution' "
                    "AND json_extract(data, '$.conversation_id')=?",
                    (identifier,),
                )
            session.connection.execute(
                "DELETE FROM records WHERE kind=? AND id=?", (kind, identifier)
            )

    def project(self, project_id: str, confirmation_name: str) -> DeletionResult:
        with self.store.transaction() as session:
            project = session.get(Project, project_id)
            if confirmation_name != project.name:
                raise DomainError("Type the project's exact name to confirm deletion.")
            self.idle(session, project_id)
            assets = session.list(Asset, project_id)
            session.connection.execute(
                "INSERT INTO deleted_resources(kind,id) VALUES('Project',?)", (project_id,)
            )
            self.remove(session, self.rows(session, project_id))
            session.connection.execute(
                "DELETE FROM records WHERE kind='Project' AND id=?", (project_id,)
            )
        return DeletionResult(
            project_id=project_id, asset_ids=[a.id for a in assets], project_deleted=True
        )

    def assets(self, project_id: str, asset_ids: list[str]) -> DeletionResult:
        identifiers = set(asset_ids)
        with self.store.transaction() as session:
            session.get(Project, project_id)
            self.idle(session, project_id)
            assets = [session.get(Asset, identifier) for identifier in identifiers]
            if any(a.project_id != project_id for a in assets):
                raise DomainError("Every selected file must belong to this project.")
            history = [
                sample
                for snapshot in session.list(Snapshot, project_id)
                for sample in snapshot.samples
            ]
            history.extend(
                sample
                for version in session.list(EvaluationSetVersion, project_id)
                for sample in version.samples
            )
            protected = {sample.asset_id for sample in history}
            protected_groups = {sample.group_id for sample in history}
            protected_images = {sample.image_key for sample in history}
            models = session.list(ModelRecord, project_id)
            protected.update(asset_id for model in models for asset_id in model.training_assets)
            protected_groups.update(group for model in models for group in model.training_groups)
            protected.update(
                a.id
                for a in assets
                if a.group_id in protected_groups or a.image_key in protected_images
            )
            if blocked := [a.name for a in assets if a.id in protected]:
                raise Conflict(
                    "These files or related cases are used by saved evaluation references, "
                    "a snapshot or a trained model: "
                    + ", ".join(sorted(blocked)[:5])
                    + ". They cannot be deleted individually without breaking learning history. "
                    "Unselect them, or delete the entire project to remove its history."
                )
            self.detach_unused_evaluation(session, project_id, assets, identifiers)
            rows = [
                row
                for row in self.rows(session, project_id)
                if (row[0] == "Asset" and row[1] in identifiers)
                or referenced_assets(row[2]) & identifiers
            ]
            session.connection.executemany(
                "INSERT INTO deleted_resources(kind,id) VALUES('Asset',?)",
                [(identifier,) for identifier in identifiers],
            )
            self.remove(session, rows)
        return DeletionResult(project_id=project_id, asset_ids=sorted(identifiers))

    @staticmethod
    def detach_unused_evaluation(
        session: Session, project_id: str, assets: list[Asset], identifiers: set[str]
    ) -> None:
        """Remove unused membership, retaining reservations for surviving patient/image aliases."""
        remaining = [a for a in session.list(Asset, project_id) if a.id not in identifiers]
        remaining_groups = {a.group_id for a in remaining}
        remaining_images = {a.image_key for a in remaining}
        removed_groups = {a.group_id for a in assets} - remaining_groups
        removed_images = {a.image_key for a in assets}
        for record in session.list(EvaluationSet, project_id):
            if removed_groups.intersection(record.cohort_groups + record.member_groups):
                session.update(
                    record.model_copy(
                        update={
                            "cohort_groups": sorted(set(record.cohort_groups) - removed_groups),
                            "member_groups": sorted(set(record.member_groups) - removed_groups),
                            "version": record.version + 1,
                        }
                    )
                )
        for reservation in session.list(EvaluationReservation, project_id):
            affected = reservation.group_id in removed_groups or removed_images.intersection(
                reservation.image_keys
            )
            if (
                affected
                and reservation.group_id not in remaining_groups
                and not remaining_images.intersection(reservation.image_keys)
            ):
                session.connection.execute(
                    "DELETE FROM records WHERE kind='EvaluationReservation' AND id=?",
                    (reservation.id,),
                )


def cleanup_storage(store: Store, artifacts: Artifacts) -> None:
    """Reclaim unreferenced blobs at startup, under the server's exclusive workspace lock.

    Requests can write a blob before committing its record. Online collection could erase
    that in-flight upload or a shared checkpoint. Follow JSON manifests before deleting any
    bytes, and conservatively retain everything if a reference cannot be inspected.
    """
    with store.transaction() as session:
        if not session.connection.execute("SELECT 1 FROM deleted_resources LIMIT 1").fetchone():
            return
        records = [json.loads(r[0]) for r in session.connection.execute("SELECT data FROM records")]
    reachable: set[str] = set()
    pending: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str) and ARTIFACT_KEY.fullmatch(value) and value not in reachable:
            reachable.add(value)
            if value.endswith(".json"):
                pending.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    try:
        visit(records)
        while pending:
            visit(artifacts.json(pending.pop()))
    except (DomainError, ValueError, OSError):
        logger.warning("Deferred file cleanup: an artifact manifest could not be inspected.")
        return
    removed = 0
    for path in artifacts.root.glob("*/*"):
        if ARTIFACT_KEY.fullmatch(path.name) and path.name not in reachable and path.is_file():
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Deferred removal of an unused artifact until the next restart.")
    if removed:
        logger.info("Reclaimed %s unused artifacts after workspace deletion.", removed)
