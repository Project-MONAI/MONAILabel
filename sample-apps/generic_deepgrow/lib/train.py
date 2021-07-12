import logging

from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
    SpatialCropForegroundd,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

from .transforms import Random2DSlice

logger = logging.getLogger(__name__)


class TrainDeepgrow(BasicTrainTask):
    def __init__(
        self,
        output_dir,
        train_datalist,
        val_datalist,
        network,
        dimension,
        roi_size,
        model_size,
        max_train_interactions,
        max_val_interactions,
        **kwargs,
    ):
        super().__init__(output_dir, train_datalist, val_datalist, network, **kwargs)

        self.dimension = dimension
        self.roi_size = roi_size
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
            ToTensord(keys=("image", "label")),
        ]

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        # Dataset preparation
        t = [
            LoadImaged(keys=("image", "label")),
            AsChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
        ]

        # Pick random slice (run more epochs to cover max slices for 2D training)
        if self.dimension == 2:
            t.append(Random2DSlice(image="image", label="label"))

        # Training
        t.extend(
            [
                AddChanneld(keys=("image", "label")),
                SpatialCropForegroundd(keys=("image", "label"), source_key="label", spatial_size=self.roi_size),
                Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
                NormalizeIntensityd(keys="image", subtrahend=208.0, divisor=388.0),
            ]
        )
        if self.dimension == 3:
            t.append(FindAllValidSlicesd(label="label", sids="sids"))
        t.extend(
            [
                AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
                AddGuidanceSignald(image="image", guidance="guidance"),
                ToTensord(keys=("image", "label")),
            ]
        )

        return t

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
