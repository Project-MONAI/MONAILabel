"""Slice boxes and 3D ROI proposals, separate from segmentation training labels."""

from typing import cast

import numpy as np

from monailabel.core.errors import DomainError
from monailabel.core.geometry import orient_plane, restore_plane
from monailabel.core.models import Asset, BoxRequest, Job, Label, RegionProposal, RoiRequest
from monailabel.core.ports import Image
from monailabel.providers.remote import RemoteSegmenter
from monailabel.providers.vision import VISION_PROVIDERS, VisionProvider
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.models import Models
from monailabel.server.storage import Artifacts, Store


class Regions:
    def __init__(self, store: Store, artifacts: Artifacts, models: Models, jobs: Jobs):
        self.store, self.artifacts, self.models, self.jobs = store, artifacts, models, jobs

    def locate(self, asset_id: str, request: BoxRequest | RoiRequest) -> Job:
        asset = self.store.get(Asset, asset_id)
        model = self.models.get(asset.project_id, request.model_id)
        if model.provider not in VISION_PROVIDERS:
            raise DomainError(
                "Free-text boxes and ROIs currently require a configured vision API model."
            )
        scope = request.slice
        end = request.end_index if isinstance(request, RoiRequest) else scope.index
        if asset.kind != "volume3d" or end >= asset.spatial_shape[scope.axis] or end < scope.index:
            raise DomainError("Select valid source-volume slices for the region.")
        if (
            not scope.window
            or not np.isfinite(scope.window).all()
            or scope.window[0] >= scope.window[1]
        ):
            raise DomainError("Boxes and ROIs require a valid intensity window from the viewer.")

        def work(context: JobContext) -> Outcome:
            # Reuse the validated vision polygon contract for localization. The temporary
            # target does not change the project's segmentation protocol or model record.
            target = Label(id=1, name=request.target, color="#ffc857")
            provider = RemoteSegmenter(
                cast(VisionProvider, model.provider),
                self.models.credentials,
            )
            volume = self.artifacts.array(asset.image_key)
            detected = 0
            extrema = []
            count = end - scope.index + 1
            for step, index in enumerate(range(scope.index, end + 1)):
                context.progress(
                    step / count,
                    f"Locating {request.target}: slice {index + 1} ({step + 1}/{count}) "
                    f"using {model.name}",
                )
                image = orient_plane(np.take(volume, index, axis=scope.axis), scope.orientation)
                low, high = scope.window or (0, 1)
                image = np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
                prediction = provider.predict(
                    cast(Image, image),
                    [Label(id=0, name="Background", color="#000000"), target],
                    request.prompt + f"\nThis image is source slice {index + 1}. "
                    "Outline the requested target on THIS image to locate its tight bounding box.",
                    model.model_copy(update={"label_ids": [0, 1]}),
                )
                coordinates = np.argwhere(restore_plane(prediction.mask, scope.orientation) == 1)
                if len(coordinates):
                    detected += 1
                    extrema.extend([coordinates.min(axis=0), coordinates.max(axis=0)])
                context.progress((step + 1) / count)
            bounds: list[list[int]] = []
            if extrema:
                points = np.asarray(extrema)
                for point, index in ((points.min(axis=0), scope.index), (points.max(axis=0), end)):
                    corner = [int(v) for v in point]
                    corner.insert(scope.axis, index)
                    bounds.append(corner)
            region = RegionProposal(
                project_id=asset.project_id,
                asset_id=asset.id,
                base_revision=asset.revision,
                model_id=model.id,
                target=request.target,
                prompt=request.prompt,
                slice=scope,
                end_index=end if isinstance(request, RoiRequest) else None,
                detected_slices=detected,
                bounds=bounds,
            )
            return Outcome({"region_id": region.id}, [region])

        return self.jobs.submit(
            "roi" if isinstance(request, RoiRequest) else "bounding_box",
            asset.project_id,
            request.model_dump(mode="json") | {"asset_id": asset_id},
            work,
        )
