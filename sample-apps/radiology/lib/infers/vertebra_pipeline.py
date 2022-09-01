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
import copy
import logging
import tempfile
from typing import Callable, Sequence

from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Resized,
    ToNumpyd,
)
from monai.transforms import SaveImage

from lib.transforms.transforms import ConcatenateROId, CropAndCreateSignald, PlaceCroppedAread
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import MergeAllPreds
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)


class InferVertebraPipeline(InferTask):
    def __init__(
            self,
            path,
            task_loc_spine: InferTask,
            task_loc_vertebra: InferTask,
            network=None,
            type=InferType.SEGMENTATION,
            labels=None,
            dimension=3,
            description="Combines three stages for vertebra segmentation",
            **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.task_loc_spine = task_loc_spine
        self.task_loc_vertebra = task_loc_vertebra

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            CropAndCreateSignald(keys="image", signal_key="signal"),
            Resized(keys=("image", "signal"), spatial_size=self.roi_size, mode=("area", "area")),
            ConcatenateROId(keys="signal"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        applied_labels = list(self.labels.values()) if isinstance(self.labels, dict) else self.labels
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=applied_labels),
            ToNumpyd(keys="pred"),
            PlaceCroppedAread(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]

    def __call__(self, request):
        # Run first stage
        req = copy.deepcopy(request)
        req["pipeline_mode"] = True
        image, label = self.task_loc_spine(req)

        # Run second stage
        req = copy.deepcopy(request)
        req["image"] = image
        req["label"] = label
        _, res_json = self.task_loc_vertebra(req)

        # Run third stage
        all_outs = {}
        all_keys = []
        result_jsons = []
        for centroid in res_json["centroids"]:
            req = copy.deepcopy(request)
            req.update({
                "centroids": [centroid],
                "pipeline_mode": True
            })

            result_file, result_json = self.run_inferer(req)
            all_keys.append(list(centroid.keys())[0])
            all_outs[list(centroid.keys())[0]] = result_file
            result_jsons.append(result_json)

        # Once all the predictions are obtained, use the label dict to reconstruct the output
        out = LoadImaged(keys=all_keys, reader="ITKReader")(all_outs)
        merged_result = MergeAllPreds(keys=all_keys)(out)

        output_file = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        merged_result.meta["filename_or_obj"] = output_file
        SaveImage(output_postfix="", output_dir="/tmp/", separate_folder=False)(merged_result)

        return output_file, result_jsons
