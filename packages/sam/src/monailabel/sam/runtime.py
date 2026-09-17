"""Local image segmentation and bidirectional medical slice propagation.

The upstream runtime is isolated here. Coordinates and returned masks
always refer to the original array; no evaluation labels enter inference.
"""

import os
from contextlib import nullcontext
from functools import cache
from importlib.resources import files
from threading import Lock
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor

from monailabel.core.models import ModelRecord, SliceScope, SpatialPrompt
from monailabel.core.ports import Image, Prediction, Progress
from monailabel.sam.weights import weights

_LOCK = Lock()


@cache
def _network(provider: str, video: bool, device: str) -> Any:
    config = OmegaConf.load(str(files(__package__).joinpath(f"resources/{provider}.yaml")))
    if video:
        config.model._target_ = "sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ"
    model = instantiate(config.model, _recursive_=True)
    model.load_state_dict(
        torch.load(weights(provider), map_location="cpu", weights_only=True)["model"]
    )
    return model.eval().to(device)


def _rgb(image: Image) -> np.ndarray[Any, np.dtype[np.uint8]]:
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return np.repeat(pixels, 3, axis=-1) if pixels.shape[-1] == 1 else pixels


def _hints(spatial: SpatialPrompt, axes: list[int]) -> tuple[Any, Any, Any]:
    # SAM uses x/y, whereas source image arrays use row/column.
    points = np.array(
        [[p.coordinates[axes[1]], p.coordinates[axes[0]]] for p in spatial.points], dtype=np.float32
    )
    labels = np.array([int(p.positive) for p in spatial.points], dtype=np.int32)
    box = (
        np.array([[p[axes[1]], p[axes[0]]] for p in spatial.box], dtype=np.float32).reshape(4)
        if spatial.box
        else None
    )
    return points if len(points) else None, labels if len(points) else None, box


class SamSegmenter:
    def predict_prompted(
        self,
        image: Image,
        label_id: int,
        model: ModelRecord,
        spatial: SpatialPrompt,
        plane: SliceScope | None,
        full_volume: bool,
        progress: Progress,
    ) -> Prediction:
        device = os.environ.get(
            "MONAILABEL_SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        # One inference at a time; predictors contain mutable image/video state.
        with _LOCK, torch.inference_mode():
            progress(0.02)
            network = _network(model.provider, full_volume, device)
            progress(0.08)
            precision = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with precision:
                if plane:
                    assert plane.window is not None
                    low, high = plane.window
                    pixels = np.clip((image - low) / (high - low), 0, 1)
                    axes = [axis for axis in range(3) if axis != plane.axis]
                else:
                    pixels, axes = image, [0, 1]
                points, labels, box = _hints(spatial, axes)
                if full_volume:
                    assert plane is not None
                    result = self._volume(network, pixels, plane, points, labels, box, progress)
                else:
                    selected = np.take(pixels, plane.index, axis=plane.axis) if plane else pixels
                    predictor = SAM2ImagePredictor(network)
                    predictor.set_image(_rgb(selected))
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        box=box,
                        multimask_output=box is None,
                    )
                    binary = masks[int(np.argmax(scores))].astype(bool)
                    result = np.zeros(image.shape[:-1], dtype=bool)
                    region: list[slice | int] = [slice(None)] * result.ndim
                    if plane:
                        region[plane.axis] = plane.index
                    result[tuple(region)] = binary
                progress(1)
                return Prediction(np.asarray(result * label_id, dtype=np.uint8))

    @staticmethod
    def _volume(
        network: Any,
        image: Image,
        plane: SliceScope,
        points: Any,
        labels: Any,
        box: Any,
        progress: Progress,
    ) -> Any:
        slices = np.moveaxis(image, plane.axis, 0)
        count, height, width = slices.shape[:3]
        size = network.image_size
        # Keep input frames and tracking state on CPU, move individual features
        # to the GPU on demand. This bounds device use for long volumes.
        frames = torch.empty((count, 3, size, size), dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        for index, frame in enumerate(slices):
            resized = np.asarray(
                PILImage.fromarray(_rgb(frame)).resize((size, size)), dtype=np.float32
            )
            frames[index] = (torch.from_numpy(resized.copy()).permute(2, 0, 1) / 255 - mean) / std
            progress(0.08 + 0.12 * (index + 1) / count)
        result = np.zeros((count, height, width), dtype=bool)
        state = network.init_state(
            frames, height, width, offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        try:
            for reverse in (False, True):
                network.reset_state(state)
                network.add_new_points_or_box(
                    state,
                    frame_idx=plane.index,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=box,
                )
                for index, _, logits in network.propagate_in_video(
                    state, start_frame_idx=plane.index, reverse=reverse
                ):
                    result[index] = (logits[0, 0] > 0).cpu().numpy()
                    done = (
                        count - plane.index + plane.index - index
                        if reverse
                        else index - plane.index + 1
                    )
                    progress(0.2 + 0.79 * min(done / count, 1))
        finally:
            network.reset_state(state)
        return np.moveaxis(result, 0, plane.axis)
