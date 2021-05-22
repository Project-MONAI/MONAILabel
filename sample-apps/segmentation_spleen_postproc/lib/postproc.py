import logging

from monai.transforms import AddChanneld, LoadImaged, ScaleIntensityRanged, Spacingd, SqueezeDimd, ToNumpyd, ToTensord

from monailabel.interfaces.tasks import InferTask, InferType
from monailabel.utils.others.post import BoundingBoxd, Restored

from .transforms import AddUnaryTermd, ApplyMONAICRFPostProcd

logger = logging.getLogger(__name__)


class SpleenCRF(InferTask):
    """
    Post Processing Task For Spleen using CRF
    """

    def __init__(
        self,
        dimension=3,
        description="A post processing step with CRF for Spleen",
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
            ScaleIntensityRanged(keys="image", a_min=-164, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            AddUnaryTermd(
                ref_prob="logits",
                unary="unary",
                scribbles="label",
                channel_dim=0,
                scribbles_bg_label=2,
                scribbles_fg_label=3,
                scale_infty=100,
                use_simplecrf=False,
            ),
            AddChanneld(keys=["image", "unary"]),
            ToTensord(keys=["image", "logits", "unary"]),
        ]

    def post_transforms(self):
        return [
            SqueezeDimd(keys=["pred", "logits"], dim=0),
            ToNumpyd(keys=["pred", "logits"]),
            Restored(keys="pred", ref_image="image"),  # undo Spacingd in pre-transform
            BoundingBoxd(keys="pred", result="result", bbox="bbox"),
        ]

    def inferer(self):
        return ApplyMONAICRFPostProcd(unary="unary", pairwise="image", post_proc_label="pred")
