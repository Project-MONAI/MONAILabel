import logging

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
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.events import DeepEditEvents
from monailabel.deepedit.handler import SaveIterationOutput
from monailabel.deepedit.interaction import DeepEditInteraction
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
        model_size=(128, 128, 128),
        max_train_interactions=20,
        max_val_interactions=10,
        save_iteration=False,
        **kwargs,
    ):
        super().__init__(output_dir, train_datalist, val_datalist, network, **kwargs)

        self.model_size = model_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions
        self.save_iteration = save_iteration

    def get_click_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                ToNumpyd(keys=("image", "label", "pred", "probability", "guidance")),
                FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy", batched=True),
                AddRandomGuidanced(
                    guidance="guidance", discrepancy="discrepancy", probability="probability", batched=True
                ),
                AddGuidanceSignald(image="image", guidance="guidance", batched=True),
                DiscardAddGuidanced(image="image", batched=True, probability=0.6),
                ToTensord(keys=("image", "label")),
            ]
        )

    def loss_function(self):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def train_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
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
        )

    def train_post_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ]
        )

    def train_handlers(self):
        handlers = super().train_handlers()
        if self.save_iteration:
            handlers.append(SaveIterationOutput(output_dir=self.output_dir))
        return handlers

    def event_names(self):
        return [DeepEditEvents]

    def val_pre_transforms(self):
        return self.train_pre_transforms()

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return DeepEditInteraction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_train_interactions,
            key_probability="probability",
            train=True,
        )

    def val_iteration_update(self):
        return DeepEditInteraction(
            transforms=self.get_click_transforms(),
            max_interactions=self.max_val_interactions,
            key_probability="probability",
            train=False,
        )
