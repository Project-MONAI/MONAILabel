"""Render a frozen review scope on its source pixels for browser inspection."""

import io

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Project, Sample, Split, VideoFrameSource
from monailabel.core.review_units import FrameScope, ReviewUnit, UnitAnnotation
from monailabel.core.video import PolygonKeyframe, TrackDocument, VideoAsset, VideoMetadata
from monailabel.server.learning_data.arrays import SampleArrays
from monailabel.server.storage import Artifacts, Store


def render(
    store: Store,
    artifacts: Artifacts,
    unit: ReviewUnit,
    revision: int,
    frame: int | None,
    overlay: bool,
) -> bytes:
    if unit.revision != revision or not unit.annotation_id:
        raise Conflict("This review item changed. Reopen its current revision.")
    annotation = store.get(UnitAnnotation, unit.annotation_id)
    project = store.get(Project, unit.project_id)
    video_frame = None
    region = None
    if isinstance(unit.scope, FrameScope):
        index = unit.scope.start if frame is None else frame
        if not unit.scope.start <= index < unit.scope.stop:
            raise DomainError("Choose a frame within this review item.")
        video = store.get(VideoAsset, unit.asset_id)
        metadata = VideoMetadata.model_validate_json(artifacts.read(video.metadata_key))
        video_frame = VideoFrameSource(
            index=index,
            timestamp=metadata.timestamps[index],
            width=video.width,
            height=video.height,
        )
    else:
        if frame is not None:
            raise DomainError("This review item is an image region.")
        region = unit.scope.region
    sample = Sample(
        asset_id=unit.asset_id,
        image_key=annotation.image_key,
        mask_key=annotation.mask_key,
        revision=annotation.source_revision,
        group_id="preview",
        split=Split.POOL,
        image_region=region,
        video_frame=video_frame,
    )
    with SampleArrays(
        artifacts,
        [sample],
        [label.id for label in project.labels],
        lambda: None,
        include_masks=video_frame is None,
    ) as arrays:
        image, mask = arrays[0]
        rgb = np.clip(image * 255, 0, 255).astype(np.uint8)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if overlay:
            for label in project.labels:
                if label.id:
                    selected = mask == label.id
                    color = np.asarray(ImageColor.getrgb(label.color), dtype=np.float32)
                    rgb[selected] = (0.55 * rgb[selected] + 0.45 * color).astype(np.uint8)
        result = Image.fromarray(rgb)
        if overlay and video_frame:
            document = TrackDocument.model_validate_json(artifacts.read(annotation.mask_key))
            draw = ImageDraw.Draw(result, "RGBA")
            colors = {label.id: ImageColor.getrgb(label.color) for label in project.labels}
            for track in document.tracks:
                key = next(
                    (k for k in reversed(track.keyframes) if k.frame <= video_frame.index), None
                )
                if key is None or key.outside:
                    continue
                following = next((k for k in track.keyframes if k.frame > video_frame.index), None)
                points = key.points if isinstance(key, PolygonKeyframe) else key.box
                if key.frame != video_frame.index and following and not following.outside:
                    end = (
                        following.points
                        if isinstance(following, PolygonKeyframe)
                        else following.box
                    )
                    if len(points) != len(end):
                        raise DomainError(
                            "Inspect this interpolated polygon in CVAT; "
                            "its vertices change between keyframes."
                        )
                    fraction = (video_frame.index - key.frame) / (following.frame - key.frame)
                    points = [a + fraction * (b - a) for a, b in zip(points, end, strict=True)]
                track_color = colors[track.label_id]
                if isinstance(key, PolygonKeyframe):
                    draw.polygon(
                        list(zip(points[::2], points[1::2], strict=True)),
                        fill=(*track_color, 95),
                        outline=(*track_color, 255),
                    )
                else:
                    draw.rectangle(points, outline=(*track_color, 255), width=2)
        result.thumbnail((1600, 1200))
        output = io.BytesIO()
        result.save(output, format="PNG")
        return output.getvalue()
