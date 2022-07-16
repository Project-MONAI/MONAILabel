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

import logging

import numpy as np
import torch
from lib.transforms.transforms import AddROI, GaussianSmoothedCentroidd, PlaceCroppedAread
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirst,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImage,
    LoadImaged,
    ScaleIntensityd,
    SpatialPadd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import LargestCCd, Restored

logger = logging.getLogger(__name__)


class InferVertebraPipeline(InferTask):
    def __init__(
        self,
        path_ver_seg,
        network_ver_seg,
        model_spine_loc: InferTask,
        model_ver_loc: InferTask,
        type=InferType.SEGMENTATION,
        dimension=3,
        description="Combines three stages for vertebra segmentation",
        spatial_size=(256, 256),
        model_size=(256, 256),
        batch_size=1,
        output_largest_cc=False,
    ):
        super().__init__(
            path=path_ver_seg,
            network=network_ver_seg,
            type=type,
            labels=None,
            dimension=dimension,
            description=description,
        )
        self.model_spine_loc = model_spine_loc
        self.model_ver_loc = model_ver_loc
        self.spatial_size = spatial_size
        self.model_size = model_size

        self.batch_size = batch_size
        self.output_largest_cc = output_largest_cc

    def pre_transforms(self, data):
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            # Include results from previous models in this transform!!
            # GetCentroidAndCropd(keys="label"),
            #
            GaussianSmoothedCentroidd(keys="image"),
            AddROI(keys="image"),
            SpatialPadd(keys="image", spatial_size=self.roi_size),
        ]

    def inferer(self, data):
        return SimpleInferer()

    def post_transforms(self, data):
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            # Loop over all vertebras in this transform
            PlaceCroppedAread(keys="pred"),
            #
            Restored(keys="pred", ref_image="image"),
        ]
        return t

    def __call__(self, request):

        #################################################
        # Run first stage
        #################################################
        result_file_first_stage, result_json_first_stage = self.model_spine_loc(request)

        # Load predicted label
        label_first_stage = LoadImage(image_only=True)(result_file_first_stage)
        label_first_stage = AsChannelFirst()(label_first_stage)
        label_first_stage = EnsureType(device=self._config.get("device"))(label_first_stage)
        logger.debug(f"Label shape: {label_first_stage.shape}")

        # Update request for second stage
        request["label"] = label_first_stage

        #################################################
        # Run second stage
        #################################################
        result_file_second_stage, result_json_second_stage = self.model_ver_loc(request)

        # Load predicted label
        label_second_stage = LoadImage(image_only=True)(result_file_second_stage)
        label_second_stage = AsChannelFirst()(label_second_stage)
        label_second_stage = EnsureType(device=self._config.get("device"))(label_second_stage)
        logger.debug(f"Label shape: {label_second_stage.shape}")

        # Update request for third stage
        request["label"] = label_second_stage

        result_file, j = super().__call__(request)
        result_json_second_stage.update(j)

        return result_file, result_json_second_stage

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):

        image = data[self.input_key]
        vertebras = data["vertebras"]  # Loop over all vertebras

        logger.debug(f"Pre processed Image shape: {image.shape}")

        batched_data = []
        batched_slices = []
        pred = np.zeros(image.shape[1:])
        logger.debug(f"Init pred: {pred.shape}")

        for vertebra_idx in vertebras:
            img = np.array([image[0][vertebra_idx], image[1][vertebra_idx], image[2][vertebra_idx]])
            # logger.info('{} => Image shape: {}'.format(vertebra_idx, img.shape))

            batched_data.append(img)
            batched_slices.append(vertebra_idx)
            if 0 < self.batch_size == len(batched_data):
                self.run_batch(super().run_inferer, batched_data, batched_slices, pred)
                batched_data = []
                batched_slices = []

        # Last batch
        if len(batched_data):
            self.run_batch(super().run_inferer, batched_data, batched_slices, pred)

        pred = pred[np.newaxis]
        logger.debug(f"Prediction: {pred.shape}; sum: {np.sum(pred)}")

        data[self.output_label_key] = pred
        return data

    def run_batch(self, run_inferer_method, batched_data, batched_slices, pred):
        bdata = {self.input_key: torch.as_tensor(batched_data)}
        outputs = run_inferer_method(bdata, False)
        for i, s in enumerate(batched_slices):
            p = torch.sigmoid(outputs[self.output_label_key][i]).detach().cpu().numpy()
            p[p > 0.5] = 1
            pred[s] = LargestCCd.get_largest_cc(p) if self.output_largest_cc else p
