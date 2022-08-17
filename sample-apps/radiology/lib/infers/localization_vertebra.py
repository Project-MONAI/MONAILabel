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
from typing import Callable, Sequence

from lib.transforms.transforms import VertebraLocalizationSegmentation
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class SimpleJsonWriter:
    def __init__(self, label="pred"):
        self.label = label

    def __call__(self, data):
        return None, data["result"]


class LocalizationVertebra(InferTask):
    """
    This provides Inference Engine for pre-trained vertebra localization (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) vertebra localization from CT image",
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
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys=("image", "first_stage_pred"), reader="ITKReader", allow_missing_keys=True),
            EnsureTyped(
                keys=("image", "first_stage_pred"), device=data.get("device") if data else None, allow_missing_keys=True
            ),
            EnsureChannelFirstd(keys=("image", "first_stage_pred"), allow_missing_keys=True),
            CropForegroundd(keys=("image", "first_stage_pred"), source_key="image", margin=10, allow_missing_keys=True),
            NormalizeIntensityd(keys="image", nonzero=True),
            GaussianSmoothd(keys="image", sigma=0.75),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.roi_size, sw_batch_size=4, overlap=0.4, padding_mode="replicate", mode="gaussian"
        )

    def post_transforms(self, data=None) -> Sequence[Callable]:
        largest_cc = False if not data else data.get("largest_cc", False)
        applied_labels = list(self.labels.values()) if isinstance(self.labels, dict) else self.labels
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]
        if largest_cc:
            t.append(KeepLargestConnectedComponentd(keys="pred", applied_labels=applied_labels))
        t.append(Restored(keys="pred", ref_image="image"))
        t.append(VertebraLocalizationSegmentation(keys="pred", result="result"))
        return t

    # This is to avoid saving the prediction
    def writer(self, data, extension=None, dtype=None):
        if data.get("result_mask", False):
            return super().writer(data, extension, dtype)
        writer = SimpleJsonWriter(label=self.output_label_key)
        return writer(data)
