"""Resolve the current review target and publish whole-source decisions to its items."""

from monailabel.core.errors import DomainError, NotFound
from monailabel.core.models import Annotation, Asset, DecisionRequest, ReviewDecision
from monailabel.core.review_units import ReviewUnit, UnitAnnotation
from monailabel.core.video import TrackAnnotation, VideoAsset
from monailabel.server.storage import Session


def target(
    session: Session, identifier: str
) -> tuple[Annotation | UnitAnnotation | TrackAnnotation, Asset | ReviewUnit | VideoAsset]:
    try:
        unit_annotation = session.get(UnitAnnotation, identifier)
    except NotFound:
        pass
    else:
        return unit_annotation, session.get(ReviewUnit, unit_annotation.unit_id)
    try:
        annotation = session.get(Annotation, identifier)
    except NotFound:
        video_annotation = session.get(TrackAnnotation, identifier)
        return video_annotation, session.get(VideoAsset, video_annotation.asset_id)
    if annotation.regions is not None:
        raise DomainError("Review each submitted region separately in the review queue.")
    return annotation, session.get(Asset, annotation.asset_id)


def decide_source_units(
    session: Session, project_id: str, asset_id: str, request: DecisionRequest, user_id: str
) -> None:
    for unit in session.list(ReviewUnit, project_id):
        if unit.asset_id == asset_id and unit.annotation_id:
            session.insert(
                ReviewDecision(
                    project_id=project_id,
                    asset_id=asset_id,
                    annotation_id=unit.annotation_id,
                    revision=unit.revision,
                    reviewer_id=user_id,
                    **request.model_dump(),
                )
            )
