"""Run real SAM 2.1 image inference and video propagation on the installed GPU.

Downloads the pinned public weights. Uses a disposable synthetic video, never a
user workspace or hosted model. This checks execution and geometry, not accuracy.
Use --frames 70 to cross the tracker's 64-frame chunk boundary.
"""

import argparse
import io
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from monailabel.core.models import ModelRecord, SpatialPrompt
from monailabel.core.video import PolygonKeyframe
from monailabel.sam.runtime import SamSegmenter, _network
from monailabel.sam.video import SamVideoTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    if not 2 <= args.frames <= 128:
        parser.error("--frames must be between 2 and 128")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this check requires a GPU.")
    # Explicitly select CUDA; an inherited CPU override must not produce a GPU pass.
    os.environ["MONAILABEL_SAM_DEVICE"] = "cuda"
    pixels = np.zeros((96, 128, 3), dtype=np.float32)
    pixels[24:72, 40:88] = (0.9, 0.6, 0.2)
    model = ModelRecord(project_id="smoke", name="SAM 2.1", provider="sam2", label_ids=[0, 1])
    prediction = SamSegmenter().predict_prompted(
        pixels,
        1,
        model,
        SpatialPrompt(box=[[24, 40], [72, 88]]),
        None,
        False,
        lambda _: None,
    )
    assert prediction.mask.shape == pixels.shape[:2]
    assert prediction.mask.dtype == np.uint8 and prediction.mask.any()
    assert set(np.unique(prediction.mask)) <= {0, 1}
    assert next(_network("sam2", False, "cuda").parameters()).is_cuda
    with tempfile.TemporaryDirectory(prefix="monailabel-sam-smoke-") as temporary:
        source = Path(temporary) / "synthetic.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=128x96:r=8",
                "-vf",
                "drawbox=x=40:y=24:w=48:h=48:color=orange:t=fill",
                "-frames:v",
                str(args.frames),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
            timeout=30,
        )
        seed = PolygonKeyframe(frame=0, points=[40, 24, 88, 24, 88, 72, 40, 72])
        result = SamVideoTracker().track(
            source, 128, 96, seed, args.frames, lambda _: None, output="polygon"
        )
        assert [key.frame for key in result.keyframes] == list(range(args.frames))
        assert result.keyframes[0] == seed
        assert set(result.masks) == set(range(args.frames))
        for content in result.masks.values():
            with Image.open(io.BytesIO(content)) as mask:
                assert mask.size == (128, 96)
        assert next(_network("sam2", True, "cuda").parameters()).is_cuda
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "architecture": platform.machine(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "synthetic_only": True,
                "image_inference": "passed",
                "video_frames": args.frames,
                "source_geometry_preserved": True,
                "seed_preserved": True,
                "hosted_api_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
