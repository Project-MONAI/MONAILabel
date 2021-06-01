from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityRanged, Spacingd

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored

from .transforms import ApplyCRFOptimisationd, ApplyGraphCutOptimisationd, MakeBIFSegUnaryd


class SpleenPostProc(InferTask):
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
                MakeBIFSegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    scale_infty=10,
                    use_simplecrf=False,
                ),
                ApplyCRFOptimisationd(unary="unary", pairwise="image", post_proc_label="pred"),
            ]
        )


class SpleenBIFSegGraphCut(SpleenPostProc):
    def __init__(
        self,
        dimension=3,
        description="A post processing step with BIFSeg GraphCut for Spleen",
    ):
        super().__init__(
            dimension=dimension,
            description=description,
        )

    def inferer(self):
        return Compose(
            [
                MakeBIFSegUnaryd(
                    image="image",
                    logits="logits",
                    scribbles="label",
                    unary="unary",
                    scribbles_bg_label=2,
                    scribbles_fg_label=3,
                    scale_infty=10,
                    use_simplecrf=True,
                ),
                ApplyGraphCutOptimisationd(unary="unary", pairwise="image", post_proc_label="pred"),
            ]
        )
