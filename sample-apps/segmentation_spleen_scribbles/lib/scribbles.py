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

from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityRanged, Spacingd

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored

from .transforms import (
    ApplyCRFOptimisationd,
    ApplyGraphCutOptimisationd,
    ApplyISegGraphCutPostProcd,
    ApplySimpleCRFOptimisationd,
    MakeISegUnaryd,
    SoftenProbSoftmax,
)


class SpleenPostProc(InferTask):
    """
    Defines a generic post processing task for Spleen segmentation.
    """

    def __init__(
        self,
        dimension,
        description,
    ):
        super().__init__(
            path=None, network=None, labels=None, type=InferType.SCRIBBLES, dimension=dimension, description=description
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "label"]),
            # at the moment optimisers are bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.5, 2.5, 5.0]),
            Spacingd(keys=["label"], pixdim=[2.5, 2.5, 5.0], mode="nearest"),
            ScaleIntensityRanged(keys="image", a_min=-300, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ]

    def post_transforms(self):
        return [
            Restored(keys="pred", ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]

    def inferer(self):
        raise NotImplementedError("inferer not implemented in base post proc class")


class SpleenISegCRF(SpleenPostProc):
    """
    Defines ISeg+CRF based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    MONAI's CRF layer is used to optimise Equation 5 from the paper, where unaries come from Equation 7
    and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with ISeg + MONAI's CRF for Spleen segmentation",
    ):
        super().__init__(dimension, description)

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "label"]),
            # at the moment optimisers are bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.5, 2.5, 5.0]),
            Spacingd(keys=["label"], pixdim=[2.5, 2.5, 5.0], mode="nearest"),
            ScaleIntensityRanged(keys="image", a_min=-300, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            SoftenProbSoftmax(logits="logits", prob="prob"),
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
                ApplyCRFOptimisationd(unary="unary", pairwise="image", post_proc_label="pred", device="cpu"),
            ]
        )


class SpleenISegGraphCut(SpleenPostProc):
    """
    Defines ISeg+GraphCut based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's GraphCut MaxFlow is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with ISeg + SimpleCRF's GraphCut for Spleen segmentation",
    ):
        super().__init__(dimension, description)

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeISegUnaryd(
                    image="image",
                    logits="logits",
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
                    lamda=10.0,
                    sigma=15.0,
                ),
            ]
        )


class SpleenInteractiveGraphCut(SpleenPostProc):
    """
    Defines ISeg+GraphCut based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's interactive GraphCut MaxFlow is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with SimpleCRF's Interactive ISeg GraphCut for Spleen segmentation",
    ):
        super().__init__(dimension, description)

    def inferer(self):
        return Compose(
            [
                ApplyISegGraphCutPostProcd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    post_proc_label="pred",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    lamda=10.0,
                    sigma=15.0,
                ),
            ]
        )


class SpleenISegSimpleCRF(SpleenPostProc):
    """
    Defines ISeg+SimpleCRF's CRF based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's CRF is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with ISeg + SimpleCRF's CRF for Spleen segmentation",
    ):
        super().__init__(dimension, description)

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeISegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                ),
                # optimiser
                ApplySimpleCRFOptimisationd(
                    unary="unary",
                    pairwise="image",
                    post_proc_label="pred",
                ),
            ]
        )
