"""Project source identities and accepted coverage, independent of model runtimes."""

from dataclasses import dataclass

from monailabel.core.models import (
    Annotation,
    Asset,
    ReviewDecision,
    Sample,
    Split,
    VideoFrameSource,
)
from monailabel.core.review_units import FrameScope, RegionScope, ReviewUnit, UnitAnnotation
from monailabel.core.video import VideoAsset, VideoMetadata
from monailabel.server.storage import Artifacts, Session


@dataclass(frozen=True)
class LearningCase:
    source: Asset | VideoAsset

    @property
    def id(self) -> str:
        return self.source.id

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def group_id(self) -> str:
        return self.source.group_id

    @property
    def split(self) -> Split:
        return self.source.split

    @property
    def image_key(self) -> str:
        """Stable decoded-image or original-video identity for leakage checks."""
        return self.source.image_key if isinstance(self.source, Asset) else self.source.source_key

    @property
    def annotation_id(self) -> str | None:
        return self.source.annotation_id


def cases(session: Session, project_id: str) -> list[LearningCase]:
    return [LearningCase(source) for source in session.list(Asset, project_id)] + [
        LearningCase(source) for source in session.list(VideoAsset, project_id)
    ]


def accepted_units(
    session: Session,
    case: LearningCase,
    required: set[int],
    decisions: dict[str, ReviewDecision],
) -> list[tuple[UnitAnnotation, ReviewDecision]]:
    result = []
    for unit in session.list(ReviewUnit, case.source.project_id):
        if unit.asset_id != case.id or not unit.annotation_id:
            continue
        decision = decisions.get(unit.annotation_id)
        if decision is None or decision.verdict != "accepted":
            continue
        annotation = session.get(UnitAnnotation, unit.annotation_id)
        if required <= set(annotation.covered_labels):
            result.append((annotation, decision))
    return result


def accepted_whole(
    session: Session,
    case: LearningCase,
    required: set[int],
    decisions: dict[str, ReviewDecision],
) -> tuple[Annotation, ReviewDecision] | None:
    if not isinstance(case.source, Asset) or not case.annotation_id:
        return None
    annotation = session.get(Annotation, case.annotation_id)
    decision = decisions.get(annotation.id)
    if (
        annotation.regions is None
        and required <= set(annotation.covered_labels)
        and decision
        and decision.verdict == "accepted"
    ):
        return annotation, decision
    return None


def unit_samples(
    session: Session,
    artifacts: Artifacts,
    case: LearningCase,
    required: set[int],
    split: Split,
    decisions: dict[str, ReviewDecision],
) -> list[Sample]:
    result = []
    for annotation, decision in accepted_units(session, case, required, decisions):
        sample = Sample(
            asset_id=case.id,
            image_key=annotation.image_key,
            mask_key=annotation.mask_key,
            revision=annotation.source_revision,
            group_id=case.group_id,
            split=split,
            decision_id=decision.id,
            annotation_id=annotation.id,
            unit_id=annotation.unit_id,
        )
        if isinstance(annotation.scope, RegionScope):
            result.append(sample.model_copy(update={"image_region": annotation.scope.region}))
        elif isinstance(annotation.scope, FrameScope) and isinstance(case.source, VideoAsset):
            metadata = VideoMetadata.model_validate_json(artifacts.read(case.source.metadata_key))
            for frame in range(annotation.scope.start, annotation.scope.stop):
                result.append(
                    sample.model_copy(
                        update={
                            "video_frame": VideoFrameSource(
                                index=frame,
                                timestamp=metadata.timestamps[frame],
                                width=case.source.width,
                                height=case.source.height,
                            )
                        }
                    )
                )
    return result
