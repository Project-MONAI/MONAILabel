from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityRanged, Spacingd

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored

from .transforms import (
    ApplyBIFSegGraphCutPostProcd,
    ApplyCRFOptimisationd,
    ApplyGraphCutOptimisationd,
    ApplySimpleCRFOptimisationd,
    MakeBIFSegUnaryd,
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
            path=None,
            network=None,
            type=InferType.POSTPROCS,
            labels=None,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
            # at the moment CRF implementation is bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.5, 2.5, 5.0]),
            Spacingd(keys=["label"], pixdim=[2.5, 2.5, 5.0], mode="nearest"),
            ScaleIntensityRanged(keys="image", a_min=-300, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ]

    def post_transforms(self):
        return [
            Restored(keys="pred", ref_image="image"),  # undo Spacingd in pre-transform
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]

    def inferer(self):
        raise NotImplementedError("inferer not implemented in base post proc class")


class SpleenBIFSegCRF(SpleenPostProc):
    """
    Defines BIFSeg+CRF based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. It incorporates the user-scribbles using
    Equation 7 on page 4 of the paper.

    It runs CRF to optimise Equation 5 from the paper (with unaries coming from Equation 7 and pairwise as input volume).
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step BIFSeg with CRF for Spleen",
    ):
        super().__init__(
            dimension=dimension,
            description=description,
        )

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeBIFSegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    scale_infty=1e6,
                ),
                # optimiser
                ApplyCRFOptimisationd(unary="unary", pairwise="image", post_proc_label="pred"),
            ]
        )


class SpleenBIFSegGraphCut(SpleenPostProc):
    """
    Defines BIFSeg+GraphCut based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. It incorporates the user-scribbles using
    Equation 7 on page 4 of the paper.

    It runs GraphCut to optimise Equation 5 from the paper (with unaries coming from Equation 7 and pairwise as input volume).
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with BIFSeg GraphCut for Spleen",
    ):
        super().__init__(
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
            # GraphCut is cheap, so use relatively smaller spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.0, 2.0, 3.0]),
            Spacingd(keys=["label"], pixdim=[2.0, 2.0, 3.0], mode="nearest"),
            # also reduce scaling range as GraphCut quantises pairwise to uint8
            ScaleIntensityRanged(keys="image", a_min=-154, a_max=154, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeBIFSegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    scale_infty=1e6,
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
    Defines BIFSeg+GraphCut based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. It incorporates the user-scribbles using
    Equation 7 on page 4 of the paper.

    It runs GraphCut to optimise Equation 5 from the paper (with unaries coming from Equation 7 and pairwise as input volume).
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with SimpleCRF's Interactive BIFSeg GraphCut for Spleen",
    ):
        super().__init__(
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
            # GraphCut is cheap, so use relatively smaller spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.0, 2.0, 3.0]),
            Spacingd(keys=["label"], pixdim=[2.0, 2.0, 3.0], mode="nearest"),
            # also reduce scaling range as GraphCut quantises pairwise to uint8
            ScaleIntensityRanged(keys="image", a_min=-154, a_max=154, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return Compose(
            [
                ApplyBIFSegGraphCutPostProcd(
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


class SpleenBIFSegSimpleCRF(SpleenPostProc):
    """
    Defines BIFSeg+SimpleCRF's CRF based post processing task for Spleen segmentation from the following paper:

    Wang, Guotai, et al. "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    IEEE transactions on medical imaging 37.7 (2018): 1562-1573. (preprint: https://arxiv.org/pdf/1710.04043.pdf)

    This task takes as input 1) original image volume 2) logits from model and 3) scribbles from user
    indicating corrections for initial segmentation from model. It incorporates the user-scribbles using
    Equation 7 on page 4 of the paper.

    It runs SimpleCRF's CRF to optimise Equation 5 from the paper
    (with unaries coming from Equation 7 and pairwise as input volume).
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with BIFSeg SimpleCRF's CRF for Spleen",
    ):
        super().__init__(
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
            # GraphCut is cheap, so use relatively smaller spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.5, 2.5, 5.0]),
            Spacingd(keys=["label"], pixdim=[2.5, 2.5, 5.0], mode="nearest"),
            # also reduce scaling range as GraphCut quantises pairwise to uint8
            ScaleIntensityRanged(keys="image", a_min=-154, a_max=154, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self):
        return Compose(
            [
                # unary term maker
                MakeBIFSegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    scale_infty=1e6,
                ),
                # optimiser
                ApplySimpleCRFOptimisationd(
                    unary="unary",
                    pairwise="image",
                    post_proc_label="pred",
                ),
            ]
        )
