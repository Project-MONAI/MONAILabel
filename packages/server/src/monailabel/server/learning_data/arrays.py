"""Bounded array access to immutable image regions and original video frames."""

import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast, overload

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from monailabel.core.errors import DomainError
from monailabel.core.geometry import region_pixels, region_slice
from monailabel.core.models import Sample
from monailabel.core.ports import IGNORE_LABEL, Image, TrainingMask
from monailabel.core.video import PolygonKeyframe, TrackDocument, TrackKeyframe
from monailabel.server.storage import Artifacts


def polygon_mask(document: TrackDocument, frame: int, width: int, height: int) -> TrainingMask:
    mask = np.zeros((height, width), dtype=np.int16)
    # A box localizes an object but does not establish its segmentation boundary.
    # Preserve those pixels as unknown unless a polygon explicitly covers them.
    for track in document.tracks:
        key = next((k for k in reversed(track.keyframes) if k.frame <= frame), None)
        if not isinstance(key, TrackKeyframe) or key.outside:
            continue
        box = key.box
        following = next((k for k in track.keyframes if k.frame > frame), None)
        if key.frame != frame and isinstance(following, TrackKeyframe) and not following.outside:
            fraction = (frame - key.frame) / (following.frame - key.frame)
            box = [a + fraction * (b - a) for a, b in zip(box, following.box, strict=True)]
        left, top = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
        right, bottom = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
        mask[top:bottom, left:right] = IGNORE_LABEL
    for track in document.tracks:
        if not isinstance(track.keyframes[0], PolygonKeyframe):
            continue
        previous = next((key for key in reversed(track.keyframes) if key.frame <= frame), None)
        if previous is None or previous.outside or not isinstance(previous, PolygonKeyframe):
            continue
        points = previous.points
        following = next((key for key in track.keyframes if key.frame > frame), None)
        if previous.frame != frame and following is not None and not following.outside:
            if not isinstance(following, PolygonKeyframe) or len(following.points) != len(points):
                raise DomainError(
                    f"Frame {frame} needs an explicit polygon keyframe before training; "
                    "its neighboring polygons have different vertices."
                )
            fraction = (frame - previous.frame) / (following.frame - previous.frame)
            points = [a + fraction * (b - a) for a, b in zip(points, following.points, strict=True)]
        canvas = PILImage.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(list(zip(points[::2], points[1::2], strict=True)), fill=1)
        selected = np.asarray(canvas, dtype=bool)
        if np.any(selected & (mask > 0) & (mask != track.label_id)):
            raise DomainError(
                f"Frame {frame} has overlapping segmentation classes. Correct it before training."
            )
        mask[selected] = track.label_id
    return mask


class SampleArrays(Sequence[tuple[Image, TrainingMask]]):
    """Decode each required frame once to disposable PNGs; keep few RGB arrays resident."""

    def __init__(
        self,
        artifacts: Artifacts,
        samples: list[Sample],
        labels: list[int],
        progress: Callable[[], None],
        *,
        include_masks: bool = True,
    ):
        self.artifacts, self.samples, self.labels, self.progress = (
            artifacts,
            samples,
            labels,
            progress,
        )
        self.temporary = tempfile.TemporaryDirectory(prefix="monailabel-training-")
        self.include_masks = include_masks
        self.frames: dict[tuple[str, int], Path] = {}
        self._read = lru_cache(maxsize=2)(self._read_sample)
        self._document = lru_cache(maxsize=2)(
            lambda key: TrackDocument.model_validate_json(self.artifacts.read(key))
        )

    def __enter__(self) -> "SampleArrays":
        try:
            grouped: dict[str, set[int]] = {}
            for sample in self.samples:
                if sample.video_frame:
                    grouped.setdefault(sample.image_key, set()).add(sample.video_frame.index)
            for number, (key, indices) in enumerate(grouped.items()):
                self._decode(key, sorted(indices), Path(self.temporary.name) / str(number))
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._read.cache_clear()
        self._document.cache_clear()
        self.temporary.cleanup()

    def _decode(self, key: str, indices: list[int], directory: Path) -> None:
        directory.mkdir()
        spans: list[tuple[int, int]] = []
        for frame in indices:
            if spans and spans[-1][1] + 1 == frame:
                spans[-1] = spans[-1][0], frame
            else:
                spans.append((frame, frame))
        script = directory / "select.txt"
        script.write_text(
            "select=" + "+".join(f"between(n\\,{start}\\,{stop})" for start, stop in spans)
        )
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-fflags",
            "+genpts",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(self.artifacts.path(key)),
            "-map",
            "0:v:0",
            "-filter_script:v",
            str(script),
            "-frames:v",
            str(len(indices)),
            "-fps_mode",
            "passthrough",
            "-threads",
            "1",
            "-start_number",
            "0",
            str(directory / "%09d.png"),
        ]
        try:
            with (
                tempfile.TemporaryFile() as errors,
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors) as process,
            ):
                try:
                    deadline = time.monotonic() + 3600
                    while process.poll() is None:
                        self.progress()
                        if time.monotonic() > deadline:
                            raise DomainError("Video frame preparation timed out.")
                        time.sleep(0.1)
                    if process.returncode:
                        raise DomainError("Could not decode the reviewed source video frames.")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        except OSError as exc:
            raise DomainError("Install FFmpeg to prepare video segmentation samples.") from exc
        for number, index in enumerate(indices):
            path = directory / f"{number:09d}.png"
            if not path.is_file():
                raise DomainError("The decoder did not return every reviewed source frame.")
            self.frames[key, index] = path

    def __len__(self) -> int:
        return len(self.samples)

    @overload
    def __getitem__(self, index: int) -> tuple[Image, TrainingMask]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[Image, TrainingMask]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[Image, TrainingMask] | list[tuple[Image, TrainingMask]]:
        if isinstance(index, slice):
            return [self._read(i) for i in range(*index.indices(len(self)))]
        return self._read(index)

    def __iter__(self) -> Iterator[tuple[Image, TrainingMask]]:
        for index in range(len(self)):
            yield self[index]

    def _read_sample(self, index: int) -> tuple[Image, TrainingMask]:
        sample = self.samples[index]
        self.progress()
        if frame := sample.video_frame:
            with PILImage.open(self.frames[sample.image_key, frame.index]) as decoded:
                image = np.asarray(decoded.convert("RGB"), dtype=np.float32) / np.float32(255)
            if image.shape != (frame.height, frame.width, 3):
                raise DomainError("Decoded frame geometry differs from the immutable source.")
            mask = (
                polygon_mask(
                    self._document(sample.mask_key), frame.index, frame.width, frame.height
                )
                if self.include_masks
                else np.zeros((frame.height, frame.width), dtype=np.uint8)
            )
        else:
            image = self.artifacts.array(sample.image_key)
            mask = self.artifacts.array(sample.mask_key)
        if sample.image_region:
            selection = region_slice(sample.image_region)
            image, mask = image[selection], mask[selection]
        mask = np.where(np.isin(mask, [IGNORE_LABEL, *self.labels]), mask, 0).astype(np.int16)
        if sample.image_region:
            mask[~region_pixels(sample.image_region)] = IGNORE_LABEL
        return cast(Image, image), mask
