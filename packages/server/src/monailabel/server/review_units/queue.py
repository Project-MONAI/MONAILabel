"""A review item is a whole source or one of its independently submitted scopes."""

from dataclasses import dataclass

from monailabel.core.models import Asset, Split
from monailabel.core.review_units import ReviewUnit
from monailabel.core.video import VideoAsset
from monailabel.server.storage import Session


@dataclass(frozen=True)
class SavedReview:
    asset_id: str
    annotation_id: str
    name: str
    split: Split
    unit_id: str | None = None


def saved_reviews(session: Session, project_id: str) -> list[SavedReview]:
    grouped: dict[str, list[ReviewUnit]] = {}
    for unit in session.list(ReviewUnit, project_id):
        grouped.setdefault(unit.asset_id, []).append(unit)
    result: list[SavedReview] = []
    sources: list[Asset | VideoAsset] = [
        *session.list(Asset, project_id),
        *session.list(VideoAsset, project_id),
    ]
    for asset in sources:
        units = grouped.get(asset.id, [])
        if units:
            result.extend(
                SavedReview(
                    asset.id,
                    unit.annotation_id,
                    f"{asset.name} · {unit.name}",
                    asset.split,
                    unit.id,
                )
                for unit in units
                if unit.annotation_id
            )
        elif asset.annotation_id:
            result.append(SavedReview(asset.id, asset.annotation_id, asset.name, asset.split))
    return result
