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

from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.multilabel.transforms import (
    AddGuidanceFromPointsCustomd,
    AddGuidanceSignalCustomd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelCustomd,
)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class DeepEditSeg(InferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        label_names=None,
        dimension=3,
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=label_names,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.label_names = label_names

    def pre_transforms(self):
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            AddChanneld(keys="image"),
            # Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
            Orientationd(keys="image", axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
            DiscardAddGuidanced(keys="image", label_names=self.label_names),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()

    def inverse_transforms(self):
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self):
        return [
            ToTensord(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]


class DeepEdit(InferTask):
    """
    This provides Inference Engine for Deepgrow over DeepEdit model.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        dimension=3,
        description="A pre-trained 3D Deepedit model based on DynUnet",
        spatial_size=(128, 128, 128),
        target_spacing=(1.5, 1.5, 2.0),
        label_names=None,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=label_names,
            dimension=dimension,
            description=description,
            config={"result_extension": [".nrrd", ".nii.gz"]},
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.label_names = label_names

    def pre_transforms(self):
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            AddChanneld(keys="image"),
            # Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear"),
            Orientationd(keys="image", axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AddGuidanceFromPointsCustomd(ref_image="image", guidance="guidance", label_names=self.label_names),
            Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
            ResizeGuidanceMultipleLabelCustomd(guidance="guidance", ref_image="image"),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance"),
            ToTensord(keys="image"),
        ]

    def inferer(self):
        return SimpleInferer()

    def inverse_transforms(self):
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self):
        return [
            ToTensord(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
