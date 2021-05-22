import logging

from monai.transforms import AddChanneld, LoadImaged, ScaleIntensityRanged, Spacingd, SqueezeDimd, ToNumpyd, ToTensord

from monailabel.interfaces.tasks import PostProcTask, PostProcType
from monailabel.utils.others.post import AddUnaryTermd, ApplyMONAICRFPostProcd, BoundingBoxd, Restored

logger = logging.getLogger(__name__)


class SpleenCRF(PostProcTask):
    """
    Basic Post Processing Task Helper
    """

    def __init__(
        self,
        method="CRF",
        type=PostProcType.POSTPROCS,
        labels="spleen",
        dimension=3,
        description="A post processing step with CRF for Spleen",
    ):
        super().__init__(method=method, type=type, labels=labels, dimension=dimension, description=description)

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
        }

    def pre_transforms(self):
        return [
            LoadImaged(keys=["image", "logits", "scribbles"]),
            AddChanneld(keys=["image", "logits", "scribbles"]),
            # at the moment CRF implementation is bottleneck taking a long time,
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=["image", "logits"], pixdim=[2.5, 2.5, 5.0]),
            Spacingd(keys=["scribbles"], pixdim=[2.5, 2.5, 5.0], mode="nearest"),
            ScaleIntensityRanged(keys="image", a_min=-164, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            AddUnaryTermd(
                ref_prob="logits",
                unary="unary",
                scribbles="scribbles",
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

    def postprocessor(self):
        return ApplyMONAICRFPostProcd(unary="unary", pairwise="image", post_proc_label="pred")
