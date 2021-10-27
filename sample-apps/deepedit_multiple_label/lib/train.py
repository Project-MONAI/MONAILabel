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

import logging

import torch
from monai.handlers import MeanDice, from_engine
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
    RandRotated,
    Resized,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.handlers import TensorBoardImageHandler
from monailabel.deepedit.interaction import InteractionMultipleLabel
from monailabel.deepedit.transforms import (
    AddGuidanceSignalCustomMultiLabeld,
    AddInitialSeedPointCustomMultiLabeld,
    FindAllValidSlicesCustomMultiLabeld,
    FindDiscrepancyRegionsCustomMultiLabeld,
    PosNegClickProbAddRandomGuidanceCustomMultiLabeld,
    SelectLabelsAbdomenDatasetd,
    SplitPredsLabeld,
)
from monailabel.tasks.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train DeepEdit model for 3D Images",
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        deepgrow_probability_train=0.5,
        deepgrow_probability_val=1.0,
        max_train_interactions=20,
        max_val_interactions=10,
        label_names=None,
        debug_mode=False,
        **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.deepgrow_probability_train = deepgrow_probability_train
        self.deepgrow_probability_val = deepgrow_probability_val
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions
        self.label_names = label_names
        self.debug_mode = debug_mode

        super().__init__(model_dir, description, **kwargs)

    def network(self):
        return self._network

    def optimizer(self):
        return torch.optim.Adam(self._network.parameters(), lr=0.0001)

    def loss_function(self):
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_click_transforms(self):
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=("image", "label", "pred")),
            # Transforms for click simulation
            FindDiscrepancyRegionsCustomMultiLabeld(keys="label", pred="pred", discrepancy="discrepancy"),
            PosNegClickProbAddRandomGuidanceCustomMultiLabeld(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
            ),
            AddGuidanceSignalCustomMultiLabeld(keys="image", guidance="guidance"),
            #
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label"), reader="nibabelreader"),
            SelectLabelsAbdomenDatasetd(keys="label", label_names=self.label_names),
            # SingleModalityLabelSanityd(keys=("image", "label"), label_names=self.label_names),
            # RandZoomd(keys=("image", "label"), prob=0.4, min_zoom=0.3, max_zoom=1.9, mode=("bilinear", "nearest")),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            RandAdjustContrastd(keys="image", gamma=6),
            RandHistogramShiftd(keys="image", num_control_points=8, prob=0.5),
            RandRotated(
                keys=("image", "label"),
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                prob=0.4,
                keep_size=True,
                mode=("bilinear", "nearest"),
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation
            FindAllValidSlicesCustomMultiLabeld(keys="label", sids="sids"),
            AddInitialSeedPointCustomMultiLabeld(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalCustomMultiLabeld(keys="image", guidance="guidance"),
            #
            ToTensord(keys=("image", "label")),
        ]

    def train_post_transforms(self):
        # FOR DICE EVALUATION
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=(True, True),
                n_classes=len(self.label_names),
            ),
            SplitPredsLabeld(keys="pred"),
            # ToCheckTransformd(keys="pred"),
        ]

    def val_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label"), reader="nibabelreader"),
            SelectLabelsAbdomenDatasetd(keys="label", label_names=self.label_names),
            # SingleModalityLabelSanityd(keys=("image", "label"), label_names=self.label_names),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image"),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation
            FindAllValidSlicesCustomMultiLabeld(keys="label", sids="sids"),
            AddInitialSeedPointCustomMultiLabeld(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalCustomMultiLabeld(keys="image", guidance="guidance"),
            #
            AsDiscreted(keys="label", to_onehot=True, num_classes=len(self.label_names)),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return InteractionMultipleLabel(
            deepgrow_probability=self.deepgrow_probability_train,
            transforms=self.get_click_transforms(),
            max_interactions=self.max_train_interactions,
            click_probability_key="probability",
            train=True,
            label_names=self.label_names,
        )

    def val_iteration_update(self):
        return InteractionMultipleLabel(
            deepgrow_probability=self.deepgrow_probability_val,
            transforms=self.get_click_transforms(),
            max_interactions=self.max_val_interactions,
            click_probability_key="probability",
            train=False,
            label_names=self.label_names,
        )

    def train_key_metric(self):
        all_metrics = dict()
        all_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]))
        for _, (key_label, _) in enumerate(self.label_names.items()):
            if key_label != "background":
                all_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label])
                )
        return all_metrics

    def val_key_metric(self):
        all_metrics = dict()
        all_metrics["val_mean_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]))
        for _, (key_label, _) in enumerate(self.label_names.items()):
            if key_label != "background":
                all_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label])
                )
        return all_metrics

    def train_handlers(self, output_dir, events_dir, evaluator, local_rank=0):
        handlers = super().train_handlers(output_dir, events_dir, evaluator, local_rank)
        return (
            handlers.append(TensorBoardImageHandler(log_dir=events_dir))
            if self.debug_mode and local_rank == 0
            else handlers
        )
