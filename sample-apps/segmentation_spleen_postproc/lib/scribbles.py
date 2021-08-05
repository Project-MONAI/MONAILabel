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
            path=None, network=None, labels=None, type=InferType.SCRIBBLE, dimension=dimension, description=description
        )

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
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


class SpleenBIFSegCRF(SpleenPostProc):
    """
    Defines BIFSeg+CRF based post processing task for Spleen segmentation from the following paper:

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
        description="A post processing step with BIFSeg + MONAI's CRF for Spleen segmentation",
    ):
        super().__init__(dimension, description)

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
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's GraphCut MaxFlow is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with BIFSeg + SimpleCRF's GraphCut for Spleen segmentation",
    ):
        super().__init__(dimension, description)

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
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's interactive GraphCut MaxFlow is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with SimpleCRF's Interactive BIFSeg GraphCut for Spleen segmentation",
    ):
        super().__init__(dimension, description)

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
    indicating corrections for initial segmentation from model. User-scribbles are incorporated using
    Equation 7 on page 4 of the paper.

    SimpleCRF's CRF is used to optimise Equation 5 from the paper,
    where unaries come from Equation 7 and pairwise is the original input volume.
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with BIFSeg + SimpleCRF's CRF for Spleen segmentation",
    ):
        super().__init__(dimension, description)

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "label"]),
            AddChanneld(keys=["image", "logits", "label"]),
            # at the moment Simple CRF implementation is bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image", "logits"], pixdim=[3.5, 3.5, 5.0]),
            Spacingd(keys=["label"], pixdim=[3.5, 3.5, 5.0], mode="nearest"),
            ScaleIntensityRanged(keys="image", a_min=-300, a_max=200, b_min=0.0, b_max=1.0, clip=True),
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
