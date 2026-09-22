"""Training-only source/selection filters and reproducible group-aware limits."""

import hashlib

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    DicomSeries,
    Job,
    Sample,
    Split,
    TrainingSampleFilter,
    TrainingSource,
)
from monailabel.server.evaluation_sets import components
from monailabel.server.learning_data.cases import cases
from monailabel.server.storage import Session


def training_sources(session: Session, project_id: str) -> list[TrainingSource]:
    from monailabel.server.dataset_downloads import sources

    assets = cases(session, project_id)
    origins = {a.id: ("uploads", "Uploaded files") for a in assets}
    catalog = {source.id: source.name for source in sources()}
    for series in session.list(DicomSeries, project_id):
        if not series.derived_from_nifti and series.asset_id in origins:
            origins[series.asset_id] = ("dicom", "DICOM imports")
    for job in session.list(Job, project_id):
        if job.kind != "dataset_import":
            continue
        template = job.request.get("template_id")
        ids = job.result.get("asset_ids")
        if isinstance(template, str) and isinstance(ids, list):
            for identifier in ids:
                if isinstance(identifier, str) and identifier in origins:
                    origins[identifier] = (f"template:{template}", catalog.get(template, template))
    # Template group IDs also cover older or partially completed imports without a job result.
    for asset in assets:
        if origins[asset.id][0].startswith("template:"):
            continue
        if asset.group_id.startswith("msd:"):
            template = asset.group_id.split(":", 2)[1]
            if template in catalog:
                origins[asset.id] = (f"template:{template}", catalog[template])
        elif asset.group_id.startswith("totalsegmentator:"):
            origins[asset.id] = ("template:totalsegmentator-ct", "TotalSegmentator CT")
    result: dict[str, TrainingSource] = {}
    for asset in assets:
        identifier, name = origins[asset.id]
        if identifier not in result:
            result[identifier] = TrainingSource(id=identifier, name=name, asset_ids=[])
        result[identifier].asset_ids.append(asset.id)
    return sorted(result.values(), key=lambda source: source.name.casefold())


def filter_samples(
    session: Session, project_id: str, samples: list[Sample], request: TrainingSampleFilter
) -> list[Sample]:
    assets = cases(session, project_id)
    allowed = {a.id for a in assets}
    if request.asset_ids is not None:
        if not set(request.asset_ids) <= allowed:
            raise DomainError("Choose training images from this project.")
        allowed &= set(request.asset_ids)
    if request.source_ids is not None:
        catalog = {source.id: source for source in training_sources(session, project_id)}
        if not set(request.source_ids) <= catalog.keys():
            raise DomainError(
                "A selected dataset/source is unavailable. Refresh the training form."
            )
        allowed &= {asset for key in request.source_ids for asset in catalog[key].asset_ids}
    training = [s for s in samples if s.split == Split.TRAIN and s.asset_id in allowed]
    if request.limit is not None:
        by_asset: dict[str, list[Sample]] = {}
        for sample in training:
            by_asset.setdefault(sample.asset_id, []).append(sample)
        groups = [
            [sample for a in group for sample in by_asset.get(a.id, [])]
            for group in components(assets).values()
        ]
        groups = [group for group in groups if group]
        groups.sort(
            key=lambda group: hashlib.sha256(min(s.image_key for s in group).encode()).digest()
        )
        training = []
        for group in groups:
            if len({s.asset_id for s in [*training, *group]}) <= request.limit:
                training.extend(group)
    if not training:
        raise DomainError(
            "No eligible training images match these filters. Check the sources, selected images, "
            "annotation status and sample limit. Evaluation images cannot be used for training."
        )
    selected = {s.asset_id for s in training}
    return [s for s in samples if s.split == Split.VALIDATION or s.asset_id in selected]
