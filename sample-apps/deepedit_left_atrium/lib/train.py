import logging

from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandZoomd,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.transforms import DiscardAddGuidanced
from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        output_dir,
        train_datalist,
        val_datalist,
        network,
        model_size=(256, 256, 128),
        max_train_interactions=20,
        max_val_interactions=10,
        **kwargs,
    ):
        super().__init__(output_dir, train_datalist, val_datalist, network, **kwargs)

        self.model_size = model_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions

    def get_click_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            DiscardAddGuidanced(image="image", probability=0.6),
            ToTensord(keys=("image", "label")),
        ]

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label")),
            RandZoomd(keys=("image", "label"), prob=0.4, min_zoom=0.3, max_zoom=1.9, mode=("bilinear", "nearest")),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            RandAdjustContrastd(keys="image", gamma=6),
            RandHistogramShiftd(keys="image", num_control_points=8, prob=0.5),
            Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
            FindAllValidSlicesd(label="label", sids="sids"),
            AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
            AddGuidanceSignald(image="image", guidance="guidance"),
            DiscardAddGuidanced(image="image", probability=0.6),
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return Interaction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_train_interactions,
            key_probability="probability",
            train=True,
        )

    def val_iteration_update(self):
        return Interaction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_val_interactions,
            key_probability="probability",
            train=False,
        )
