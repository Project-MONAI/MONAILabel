"""Review decisions refer to immutable annotation revisions, never predictions."""

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    Annotation,
    Asset,
    BatchReviewRequest,
    DecisionRequest,
    ReviewDecision,
    User,
)
from monailabel.server.model_splits import refresh_percentage_sets
from monailabel.server.storage import Store


class Reviews:
    def __init__(self, store: Store):
        self.store = store

    def decide(self, annotation_id: str, request: DecisionRequest, user: User) -> ReviewDecision:
        with self.store.transaction() as session:
            annotation = session.get(Annotation, annotation_id)
            asset = session.get(Asset, annotation.asset_id)
            if asset.annotation_id != annotation.id:
                raise Conflict("This annotation was superseded. Review the current revision.")
            decision = ReviewDecision(
                project_id=asset.project_id,
                asset_id=asset.id,
                annotation_id=annotation.id,
                revision=annotation.revision,
                reviewer_id=user.id,
                **request.model_dump(),
            )
            session.insert(decision)
            if request.verdict == "accepted":
                refresh_percentage_sets(session, asset.project_id)
        return decision

    def decide_batch(
        self, project_id: str, request: BatchReviewRequest, user: User
    ) -> list[ReviewDecision]:
        with self.store.transaction() as session:
            decisions = {d.annotation_id: d for d in session.list(ReviewDecision, project_id)}
            result = []
            seen = set()
            for item in request.items:
                identifier = item.annotation_id
                if identifier in seen:
                    continue
                seen.add(identifier)
                annotation = session.get(Annotation, identifier)
                asset = session.get(Asset, annotation.asset_id)
                if asset.project_id != project_id:
                    raise DomainError("Every review must belong to this project.")
                if asset.annotation_id != annotation.id:
                    raise Conflict("An annotation changed. Refresh reviews and try again.")
                previous = decisions.get(identifier)
                if (previous.id if previous else None) != item.decision_id:
                    raise Conflict("A review decision changed. Refresh reviews and try again.")
                if (previous.verdict if previous else "pending") == request.verdict:
                    continue
                decision = ReviewDecision(
                    project_id=project_id,
                    asset_id=asset.id,
                    annotation_id=annotation.id,
                    revision=annotation.revision,
                    reviewer_id=user.id,
                    verdict=request.verdict,
                    comment=request.comment,
                )
                session.insert(decision)
                result.append(decision)
            if result and request.verdict == "accepted":
                refresh_percentage_sets(session, project_id)
            return result
