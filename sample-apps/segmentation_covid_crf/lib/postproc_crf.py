import logging
from monailabel.utils.postproc import PostProcessingTask, PostProcType
from lib.transforms import AddUnaryTermd, ApplyCRFPostProcd

from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Spacingd,
    ToNumpyd,
    SqueezeDimd,
    ScaleIntensityRanged,
    ToTensord,
)

from monailabel.utils.others.post import Restored, BoundingBoxd

logger = logging.getLogger(__name__)


class MyCRF(PostProcessingTask):
    """
    Basic Post Processing Task Helper
    """

    def __init__(self, method='CRF', type=PostProcType.POSTPROCS, labels=["spleen"], dimension=3, description='A post processing step with CRF for Spleen'):
        super().__init__(
                    method=method,
                    type=type,
                    labels=labels,
                    dimension=dimension,
                    description=description
                )

    def info(self):
        return {
            "type": self.type,
            "labels": self.labels,
            "dimension": self.dimension,
            "description": self.description,
        }

    def pre_transforms(self):
        return [
            LoadImaged(keys=['image', 'logits', 'scribbles']),
            AddChanneld(keys=['image', 'logits', 'scribbles']),
            
            # at the moment CRF implementation is bottleneck taking a long time, 
            # therefore scaling non-isotropic with big spacing
            Spacingd(keys=['image', 'logits'], pixdim=[2.0, 2.0, 2.0]),
            Spacingd(keys=['scribbles'], pixdim=[2.0, 2.0, 2.0], mode='nearest'),

            # ScaleIntensityRanged(keys='image', a_min=-164, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            AddUnaryTermd(ref_prob='logits', unary="unary", scribbles="scribbles", channel_dim=0, sc_background_label=2, sc_foreground_label=3, scale_infty=100, use_simplecrf=False),
            AddChanneld(keys=['image', 'unary']),
            ToTensord(keys=['image', 'logits', 'unary'])

        ]

    def post_transforms(self):
        return [
            SqueezeDimd(keys=['pred', 'logits'], dim=0),
            ToNumpyd(keys=['pred', 'logits']),
            Restored(keys='pred', ref_image='image'), # undo Spacingd in pre-transform
            BoundingBoxd(keys='pred', result='result', bbox='bbox'),
        ]

    def postprocessor(self):
        """
        Provide postprocessor Class

            For Example::
                return ApplyCRFPostProcd(unary='unary', pairwise='image', post_proc_label='pred', use_simplecrf=False)

        """
        return ApplyCRFPostProcd(unary='unary', pairwise='image', post_proc_label='pred', use_simplecrf=False)

