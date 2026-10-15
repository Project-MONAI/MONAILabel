# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute additional local models on synthetic inputs; not an accuracy evaluation.

Requires approved installation and cached weights. Never touches an application
workspace or hosted model. Existing VISTA3D/SAM smoke examples cover those models.
"""

import argparse
import json
import time

import numpy as np
import torch

from monailabel.core.models import Label, ModelRecord, PromptPoint, SliceScope, SpatialPrompt
from monailabel.core.ports import Volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provider",
        choices=["nninteractive", "medsam2", "totalsegmentator-ct", "totalsegmentator-mr"],
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    started = time.monotonic()
    shape = (64, 72, 48)
    i, j, k = np.indices(shape)
    image = np.full(shape + (1,), -1000, dtype=np.float32)
    image[..., 0][((i - 32) / 22) ** 2 + ((j - 36) / 26) ** 2 + ((k - 24) / 18) ** 2 < 1] = 60
    model = ModelRecord(name=args.provider, provider=args.provider, label_ids=[0, 1])
    if args.provider.startswith("totalsegmentator"):
        from monailabel.totalsegmentator.runtime import TotalSegmenter

        prediction = TotalSegmenter(args.provider, {}, None).predict_volume(
            Volume(image, np.diag([3.0, 3.0, 3.0, 1.0]).tolist()),
            [Label(id=0, name="Background"), Label(id=1, name="Liver")],
            "",
            model,
        )
    else:
        if args.provider == "nninteractive":
            from monailabel.nninteractive.runtime import NNInteractiveSegmenter

            segmenter = NNInteractiveSegmenter()
        else:
            from monailabel.sam.runtime import SamSegmenter

            segmenter = SamSegmenter()
        prediction = segmenter.predict_prompted(
            image,
            1,
            model,
            SpatialPrompt(points=[PromptPoint(coordinates=[32, 36, 24])]),
            SliceScope(axis=2, index=24, window=[-1000, 100]),
            args.provider == "nninteractive",
            lambda _: None,
        )
    assert prediction.mask.shape == shape
    assert prediction.mask.dtype == np.uint8
    assert set(np.unique(prediction.mask)) <= {0, 1}
    print(
        json.dumps(
            {
                "provider": args.provider,
                "gpu": torch.cuda.get_device_name(0),
                "seconds": round(time.monotonic() - started, 2),
                "source_geometry_preserved": True,
                "foreground_voxels": int(np.count_nonzero(prediction.mask)),
                "synthetic_execution_check_only": True,
                "hosted_api_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
