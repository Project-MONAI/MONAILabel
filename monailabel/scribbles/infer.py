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

from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.scribbles.transforms import (
    AddBackgroundScribblesFromROId,
    ApplyGraphCutOptimisationd,
    MakeISegUnaryd,
    MakeLikelihoodFromScribblesHistogramd,
)
from monailabel.utils.others.post import BoundingBoxd, Restored


class GenericISegGraphcutModelFree(InferTask):
    """
    Defines model-free Graphcut task for Generic segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume and 2) scribbles from user
    indicating foreground and background regions. A likelihood volume is generated using histogram method.
    User-scribbles are incorporated using Equation 7 on page 4 of the paper.

    SimpleCRF's Graphcut layer is used to optimise Equation 5 from the paper, where unaries come from Equation 7
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with model-free Graphcut for Generic segmentation",
        intensity_range=(-300, 200),
        pix_dim=(2.5, 2.5, 5.0),
    ):
        super().__init__(
            path=None, network=None, labels=None, type=InferType.SCRIBBLES, dimension=dimension, description=description
        )
        self.intensity_range = intensity_range
        self.pix_dim = pix_dim

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            AddBackgroundScribblesFromROId(scribbles="label", scribbles_bg_label=2, scribbles_fg_label=3),
            # at the moment optimisers are bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image"], pixdim=self.pix_dim, mode="bilinear"),
            Spacingd(keys=["label"], pixdim=self.pix_dim, mode="nearest"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            MakeLikelihoodFromScribblesHistogramd(
                image="image", scribbles="label", post_proc_label="prob", scribbles_bg_label=2, scribbles_fg_label=3
            ),
        ]

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeISegUnaryd(
                    image="image",
                    logits="prob",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                ),
                # optimiser
                ApplyGraphCutOptimisationd(
                    unary="unary",
                    pairwise="image",
                    post_proc_label="pred",
                    lamda=1.0,
                    sigma=0.1,
                ),
            ]
        )

    def post_transforms(self):
        return [
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]
