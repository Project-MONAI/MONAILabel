"""Bounded SAM 2 propagation from tool boxes or masks on source frames."""

import io
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw

from monailabel.core.errors import DomainError
from monailabel.core.ports import Progress, VideoTrackingResult
from monailabel.core.video import PolygonKeyframe, TrackKeyframe, VideoKeyframe
from monailabel.providers.video_geometry import mask_png, polygon_from_mask
from monailabel.sam.runtime import _LOCK, _network


class SamVideoTracker:
    def track(
        self,
        source: Path,
        width: int,
        height: int,
        seed: VideoKeyframe,
        frame_count: int,
        progress: Progress,
        output: str = "box",
        seed_mask: bytes | None = None,
    ) -> VideoTrackingResult:
        device = os.environ.get(
            "MONAILABEL_SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        result: list[VideoKeyframe] = []
        masks: dict[int, bytes] = {}
        complex_frames = 0
        points_total = 0
        carry = None
        if seed_mask:
            with PILImage.open(io.BytesIO(seed_mask)) as seed_image:
                if seed_image.size != (width, height):
                    raise DomainError("Seed mask dimensions do not match the source frame.")
                carry = np.asarray(seed_image.convert("L")) > 0
        elif isinstance(seed, PolygonKeyframe):
            image = PILImage.new("L", (width, height), 0)
            ImageDraw.Draw(image).polygon(
                list(zip(seed.points[::2], seed.points[1::2], strict=True)), fill=1
            )
            carry = np.asarray(image) > 0
        with _LOCK, torch.inference_mode():
            progress(0.01)
            network = _network("sam2", True, device)
            size = network.image_size
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            last: VideoKeyframe = seed
            end = seed.frame + frame_count
            chunk_start = seed.frame
            while chunk_start < end:
                # Overlap one frame and carry its exact mask to preserve the same target.
                count = min(64, end - chunk_start)
                frames = torch.empty((count, 3, size, size), dtype=torch.float32)
                command = [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-fflags",
                    "+genpts",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"select=between(n\\,{chunk_start}\\,{chunk_start + count - 1}),"
                    f"scale={size}:{size}",
                    "-frames:v",
                    str(count),
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
                with subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                ) as process:
                    try:
                        assert process.stdout is not None
                        for index in range(count):
                            raw = process.stdout.read(size * size * 3)
                            if len(raw) != size * size * 3:
                                raise DomainError(
                                    "Could not decode the requested source video frames."
                                )
                            pixels = (
                                np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3).copy()
                            )
                            frames[index] = (
                                torch.from_numpy(pixels).permute(2, 0, 1) / 255 - mean
                            ) / std
                            progress(0.01 + 0.98 * len(result) / frame_count)
                        if process.wait(timeout=30):
                            raise DomainError("Could not decode the requested source video frames.")
                    finally:
                        if process.poll() is None:
                            process.kill()
                precision = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                    else nullcontext()
                )
                with precision:
                    state = network.init_state(
                        frames, height, width, offload_video_to_cpu=True, offload_state_to_cpu=True
                    )
                    try:
                        if carry is not None:
                            network.add_new_mask(state, frame_idx=0, obj_id=1, mask=carry)
                        else:
                            network.add_new_points_or_box(
                                state,
                                frame_idx=0,
                                obj_id=1,
                                box=np.asarray(seed.box, dtype=np.float32),
                            )
                        for index, _, logits in network.propagate_in_video(
                            state, start_frame_idx=0
                        ):
                            if chunk_start != seed.frame and index == 0:
                                continue
                            number = chunk_start + index
                            binary = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                            # Keep a supplied segmentation exact on the first frame.
                            if number == seed.frame and carry is not None:
                                binary = carry.astype(np.uint8)
                            carry = binary > 0
                            ys, xs = np.where(binary)
                            key: VideoKeyframe
                            if output == "polygon":
                                masks[number] = mask_png(binary)
                                if len(xs):
                                    last, complex_shape = polygon_from_mask(
                                        binary, number, seed.occluded
                                    )
                                    complex_frames += complex_shape
                                elif not isinstance(last, PolygonKeyframe):
                                    raise DomainError(
                                        "SAM 2 could not segment the starting tool. Refine the box."
                                    )
                                key = last.model_copy(
                                    update={"frame": number, "outside": not len(xs)}
                                )
                                assert isinstance(key, PolygonKeyframe)
                                if number == seed.frame and isinstance(seed, PolygonKeyframe):
                                    key = seed
                                points_total += len(key.points)
                                if points_total > 1_000_000:
                                    raise DomainError(
                                        "Polygon proposal exceeds the geometry limit. "
                                        "Use a shorter frame range."
                                    )
                            else:
                                box = (
                                    [
                                        float(xs.min()),
                                        float(ys.min()),
                                        float(xs.max() + 1),
                                        float(ys.max() + 1),
                                    ]
                                    if len(xs)
                                    else last.box
                                )
                                key = TrackKeyframe(
                                    frame=number,
                                    box=box,
                                    outside=not len(xs),
                                    occluded=seed.occluded,
                                )
                                if number == seed.frame and isinstance(seed, TrackKeyframe):
                                    key = seed
                            last = key
                            result.append(key)
                            progress(0.01 + 0.98 * len(result) / frame_count)
                    finally:
                        network.reset_state(state)
                if chunk_start + count == end:
                    break
                chunk_start += count - 1
        warnings = (
            [
                f"{complex_frames} frames contain holes or disconnected regions. "
                "Polygons show the largest outer outline; "
                "download the original masks for all pixels."
            ]
            if complex_frames
            else []
        )
        return VideoTrackingResult(result, masks, warnings)
