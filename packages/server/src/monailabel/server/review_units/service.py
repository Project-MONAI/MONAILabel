"""Publish scoped revisions without invalidating unrelated accepted work."""

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.geometry import region_pixels, region_slice
from monailabel.core.models import Annotation, Asset, ImageRegion, ReviewDecision, User
from monailabel.core.review_units import (
    FrameScope,
    RegionScope,
    ReviewUnit,
    UnitAnnotation,
    UnitDecision,
    UnitScope,
)
from monailabel.core.video import PolygonKeyframe, TrackAnnotation, TrackDocument, VideoAsset
from monailabel.server.review_units.video import changed_frames
from monailabel.server.storage import Artifacts, Session, Store


def overlaps(first: ImageRegion, second: ImageRegion) -> bool:
    x, y = max(first.x, second.x), max(first.y, second.y)
    stop_x = min(first.x + first.width, second.x + second.width)
    stop_y = min(first.y + first.height, second.y + second.height)
    if stop_x <= x or stop_y <= y:
        return False
    a = region_pixels(first)[y - first.y : stop_y - first.y, x - first.x : stop_x - first.x]
    b = region_pixels(second)[y - second.y : stop_y - second.y, x - second.x : stop_x - second.x]
    return bool(np.any(a & b))


class ReviewUnits:
    def __init__(self, store: Store, artifacts: Artifacts):
        self.store, self.artifacts = store, artifacts

    def migrate_videos(self) -> None:
        """Expose existing submissions as review items, preserving their human decisions."""
        with self.store.transaction() as session:
            covered = {unit.asset_id for unit in session.list(ReviewUnit)}
            decisions = {d.annotation_id: d for d in session.list(ReviewDecision)}
            for asset in session.list(VideoAsset):
                if not asset.annotation_id or asset.id in covered:
                    continue
                annotation = session.get(TrackAnnotation, asset.annotation_id)
                document = TrackDocument.model_validate_json(
                    self.artifacts.read(annotation.tracks_key)
                )
                units = self.submit_frames(session, asset, annotation, document, TrackDocument())
                decision = decisions.get(annotation.id)
                if decision:
                    for unit in units:
                        session.insert(
                            ReviewDecision(
                                project_id=asset.project_id,
                                asset_id=asset.id,
                                annotation_id=unit.id,
                                revision=unit.revision,
                                reviewer_id=decision.reviewer_id,
                                verdict=decision.verdict,
                                comment=decision.comment,
                                created_at=decision.created_at,
                            )
                        )

    @staticmethod
    def for_asset(session: Session, project_id: str, asset_id: str) -> list[ReviewUnit]:
        return [u for u in session.list(ReviewUnit, project_id) if u.asset_id == asset_id]

    @staticmethod
    def publish(
        session: Session,
        unit: ReviewUnit,
        scope: UnitScope,
        source: Annotation | TrackAnnotation,
        image_key: str,
        mask_key: str,
        label_ids: list[int],
        created_by: str,
    ) -> UnitAnnotation:
        annotation = UnitAnnotation(
            project_id=source.project_id,
            asset_id=source.asset_id,
            unit_id=unit.id,
            revision=unit.revision + 1,
            source_revision=source.revision,
            source_annotation_id=source.id,
            scope=scope,
            covered_labels=label_ids,
            image_key=image_key,
            mask_key=mask_key,
            created_by=created_by,
        )
        session.insert(annotation)
        session.update(
            unit.model_copy(
                update={
                    "annotation_id": annotation.id,
                    "revision": annotation.revision,
                    "label_ids": label_ids,
                }
            )
        )
        return annotation

    def submit_regions(
        self, session: Session, asset: Asset, annotation: Annotation, regions: list[ImageRegion]
    ) -> list[UnitAnnotation]:
        if asset.kind != "image2d":
            raise DomainError("Region review requires a 2D source image.")
        if len({region.model_dump_json() for region in regions}) != len(regions):
            raise DomainError("Submit each region only once.")
        units = self.for_asset(session, asset.project_id, asset.id)
        mask = self.artifacts.array(annotation.mask_key)
        result = []
        for region in regions:
            if (
                region.y + region.height > asset.spatial_shape[0]
                or region.x + region.width > asset.spatial_shape[1]
            ):
                raise DomainError("The review region lies outside the source image.")
            scope = RegionScope(region=region)
            unit = next((u for u in units if u.scope == scope), None)
            if unit is None:
                conflict = next(
                    (
                        u
                        for u in units
                        if isinstance(u.scope, RegionScope) and overlaps(u.scope.region, region)
                    ),
                    None,
                )
                if conflict:
                    raise Conflict(
                        f"This region overlaps {conflict.name}. Reopen that region to revise it, "
                        "or select a separate region."
                    )
                unit = ReviewUnit(
                    project_id=asset.project_id,
                    asset_id=asset.id,
                    name=f"Region {len(units) + 1}",
                    scope=scope,
                    label_ids=annotation.covered_labels,
                )
                session.insert(unit)
                units.append(unit)
            if unit.annotation_id:
                previous = session.get(UnitAnnotation, unit.annotation_id)
                footprint = region_pixels(region)
                before = self.artifacts.array(previous.mask_key)[region_slice(region)][footprint]
                after = mask[region_slice(region)][footprint]
                if previous.covered_labels == annotation.covered_labels and np.array_equal(
                    before, after
                ):
                    result.append(previous)
                    continue
            result.append(
                self.publish(
                    session,
                    unit,
                    scope,
                    annotation,
                    asset.image_key,
                    annotation.mask_key,
                    annotation.covered_labels,
                    annotation.reviewer,
                )
            )
        return result

    def submit_frames(
        self,
        session: Session,
        asset: VideoAsset,
        annotation: TrackAnnotation,
        document: TrackDocument,
        previous: TrackDocument,
    ) -> list[UnitAnnotation]:
        changed = changed_frames(previous, document, asset.frames)
        if not changed:
            return []
        units = self.for_asset(session, asset.project_id, asset.id)
        selected = []
        for unit in units:
            if isinstance(unit.scope, FrameScope) and any(
                unit.scope.start <= frame < unit.scope.stop for frame in changed
            ):
                selected.append(unit)
                changed = {
                    frame for frame in changed if not unit.scope.start <= frame < unit.scope.stop
                }
        ranges: list[FrameScope] = []
        for frame in sorted(changed):
            if ranges and ranges[-1].stop == frame:
                ranges[-1] = FrameScope(start=ranges[-1].start, stop=frame + 1)
            else:
                ranges.append(FrameScope(start=frame, stop=frame + 1))
        for scope in ranges:
            unit = ReviewUnit(
                project_id=asset.project_id,
                asset_id=asset.id,
                name=f"Frames {scope.start}–{scope.stop - 1}",
                scope=scope,
                label_ids=[],
            )
            session.insert(unit)
            selected.append(unit)
        result = []
        for unit in selected:
            if not isinstance(unit.scope, FrameScope):
                continue
            labels = sorted(
                {0, *unit.label_ids}
                | {
                    track.label_id
                    for track in document.tracks
                    if any(
                        key.frame < unit.scope.stop
                        and (
                            track.keyframes[index + 1].frame
                            if index + 1 < len(track.keyframes)
                            else asset.frames
                        )
                        > unit.scope.start
                        and not key.outside
                        and isinstance(key, PolygonKeyframe)
                        for index, key in enumerate(track.keyframes)
                    )
                }
            )
            result.append(
                self.publish(
                    session,
                    unit,
                    unit.scope,
                    annotation,
                    asset.source_key,
                    annotation.tracks_key,
                    labels,
                    annotation.created_by,
                )
            )
        return result

    def decide(self, identifier: str, request: UnitDecision, user: User) -> ReviewDecision:
        with self.store.transaction() as session:
            unit = session.get(ReviewUnit, identifier)
            if unit.revision != request.base_revision or not unit.annotation_id:
                raise Conflict("This review item changed. Reopen its current revision.")
            previous = next(
                (
                    d
                    for d in reversed(session.list(ReviewDecision, unit.project_id))
                    if d.annotation_id == unit.annotation_id
                ),
                None,
            )
            if (previous.id if previous else None) != request.decision_id:
                raise Conflict("The review decision changed. Refresh before reviewing.")
            decision = ReviewDecision(
                project_id=unit.project_id,
                asset_id=unit.asset_id,
                annotation_id=unit.annotation_id,
                revision=unit.revision,
                reviewer_id=user.id,
                verdict=request.verdict,
                comment=request.comment,
            )
            session.insert(decision)
            if request.verdict == "accepted":
                from monailabel.server.model_splits import refresh_percentage_sets

                refresh_percentage_sets(session, unit.project_id)
        return decision

    def reconcile_regions(self, session: Session, asset: Asset, annotation: Annotation) -> None:
        regions = [
            unit.scope.region
            for unit in self.for_asset(session, asset.project_id, asset.id)
            if isinstance(unit.scope, RegionScope)
        ]
        if regions:
            self.submit_regions(session, asset, annotation, regions)
