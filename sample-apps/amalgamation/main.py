# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Dict

from lib import Deepgrow, LiverAndTumor, Spleen

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.scribbles.infer import HistogramBasedGraphCut
from monailabel.tasks.infer.deepgrow_pipeline import InferDeepgrowPipeline

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="DeepLearning",
            description="Multiple DeepLearning models",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        ngc_path = "https://api.ngc.nvidia.com/v2/models/nvidia/med"
        models = {
            "clara_spleen": {
                "path": os.path.join(self.model_dir, "clara_spleen.pt"),
                "uri": f"{ngc_path}/clara_pt_spleen_ct_segmentation/versions/1/files/models/model.ts",
            },
            "clara_liver_tumor": {
                "path": os.path.join(self.model_dir, "clara_liver_tumor.pt"),
                "uri": f"{ngc_path}/clara_pt_liver_and_tumor_ct_segmentation/versions/1/files/models/model.ts",
            },
            "clara_deepgrow_2d": {
                "path": os.path.join(self.model_dir, "clara_deepgrow_2d.pt"),
                "uri": f"{ngc_path}/clara_pt_deepgrow_2d_annotation/versions/1/files/models/model.ts",
            },
            "clara_deepgrow_3d": {
                "path": os.path.join(self.model_dir, "clara_deepgrow_3d.pt"),
                "uri": f"{ngc_path}/clara_pt_deepgrow_3d_annotation/versions/1/files/models/model.ts",
            },
        }
        self.download([(v["path"], v["uri"]) for v in models.values()])

        deepgrow_labels = [
            "spleen",
            "right kidney",
            "left kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferior vena cava",
            "portal vein and splenic vein",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
        ]

        infers = {
            "clara_spleen": Spleen(models["clara_spleen"]["path"]),
            "clara_liver_tumor": LiverAndTumor(models["clara_liver_tumor"]["path"]),
            "clara_deepgrow_2d": Deepgrow(models["clara_deepgrow_2d"]["path"], dimension=2, labels=deepgrow_labels),
            "clara_deepgrow_3d": Deepgrow(models["clara_deepgrow_3d"]["path"], dimension=3, labels=deepgrow_labels),
            "Histogram+GraphCut": HistogramBasedGraphCut(
                intensity_range=(-300, 200, 0.0, 1.0, True), pix_dim=(2.5, 2.5, 5.0), lamda=1.0, sigma=0.1
            ),
        }

        infers["clara_deepgrow_pipeline"] = InferDeepgrowPipeline(
            path=models["clara_deepgrow_2d"]["path"],
            model_3d=infers["clara_deepgrow_3d"],
            description="Combines Clara Deepgrow 2D and 3D models",
        )
        return infers
