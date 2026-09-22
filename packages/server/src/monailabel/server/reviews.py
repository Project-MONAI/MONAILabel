"""Review decisions refer to immutable annotation revisions, never predictions."""

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    BatchReviewRequest,
    DecisionRequest,
    ReviewDecision,
    User,
)
from monailabel.core.review_units import UnitAnnotation
from monailabel.server.model_splits import refresh_percentage_sets
from monailabel.server.review_units.decisions import decide_source_units, target
from monailabel.server.storage import Store


class Reviews:
    def __init__(self, store: Store):
        self.store = store

    def decide(self, annotation_id: str, request: DecisionRequest, user: User) -> ReviewDecision:
        with self.store.transaction() as session:
            annotation, asset = target(session, annotation_id)
            if asset.annotation_id != annotation.id:
                raise Conflict("This annotation was superseded. Review the current revision.")
            decision = ReviewDecision(
                project_id=asset.project_id,
                asset_id=annotation.asset_id,
                annotation_id=annotation.id,
                revision=annotation.revision,
                reviewer_id=user.id,
                **request.model_dump(),
            )
            session.insert(decision)
            if not isinstance(annotation, UnitAnnotation):
                decide_source_units(session, asset.project_id, asset.id, request, user.id)
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
                annotation, asset = target(session, identifier)
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
                    asset_id=annotation.asset_id,
                    annotation_id=annotation.id,
                    revision=annotation.revision,
                    reviewer_id=user.id,
                    verdict=request.verdict,
                    comment=request.comment,
                )
                session.insert(decision)
                if not isinstance(annotation, UnitAnnotation):
                    decide_source_units(
                        session,
                        project_id,
                        asset.id,
                        DecisionRequest(verdict=request.verdict, comment=request.comment),
                        user.id,
                    )
                result.append(decision)
            if result and request.verdict == "accepted":
                refresh_percentage_sets(session, project_id)
            return result
