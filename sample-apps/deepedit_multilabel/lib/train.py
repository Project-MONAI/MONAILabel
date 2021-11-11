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
import glob
import logging
import os

import torch
from monai.handlers import MeanDice, from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.handlers import TensorBoardImageHandler
from monailabel.deepedit.multilabel.interaction import Interaction
from monailabel.deepedit.multilabel.transforms import (
    AddGuidanceSignalCustomd,
    AddInitialSeedPointCustomd,
    FindAllValidSlicesCustomd,
    FindDiscrepancyRegionsCustomd,
    PosNegClickProbAddRandomGuidanceCustomd,
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
        deepgrow_probability_train=0.4,
        deepgrow_probability_val=1.0,
        max_train_interactions=10,
        max_val_interactions=5,
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
        # return torch.optim.AdamW(self._network.parameters(), lr=1e-4, weight_decay=1e-5)

    def loss_function(self):
        # return DiceLoss(to_onehot_y=True, softmax=True)
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def get_click_transforms(self):
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys=("image", "label", "pred")),
            # Transforms for click simulation
            FindDiscrepancyRegionsCustomd(keys="label", pred="pred", discrepancy="discrepancy"),
            PosNegClickProbAddRandomGuidanceCustomd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
            ),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance"),
            #
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self):
        return [
            LoadImaged(keys=("image", "label"), reader="nibabelreader"),
            SelectLabelsAbdomenDatasetd(keys="label", label_names=self.label_names),
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=("image", "label"),
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys="image",
                offsets=0.10,
                prob=0.50,
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation
            FindAllValidSlicesCustomd(keys="label", sids="sids"),
            AddInitialSeedPointCustomd(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance"),
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
            AddChanneld(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys=("image"),
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation
            FindAllValidSlicesCustomd(keys="label", sids="sids"),
            AddInitialSeedPointCustomd(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance"),
            #
            AsDiscreted(keys="label", to_onehot=True, num_classes=len(self.label_names)),
            ToTensord(keys=("image", "label")),
        ]

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_train,
            transforms=self.get_click_transforms(),
            max_interactions=self.max_train_interactions,
            click_probability_key="probability",
            train=True,
            label_names=self.label_names,
        )

    def val_iteration_update(self):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_val,
            transforms=self.get_click_transforms(),
            max_interactions=self.max_val_interactions,
            click_probability_key="probability",
            train=False,
            label_names=self.label_names,
        )

    def train_key_metric(self):
        all_metrics = dict()
        all_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]), include_background=False)
        for _, (key_label, _) in enumerate(self.label_names.items()):
            if key_label != "background":
                all_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                )
        return all_metrics

    def val_key_metric(self):
        all_metrics = dict()
        all_metrics["val_mean_dice"] = MeanDice(
            output_transform=from_engine(["pred", "label"]), include_background=False
        )
        for _, (key_label, _) in enumerate(self.label_names.items()):
            if key_label != "background":
                all_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                )
        return all_metrics

    def partition_datalist(self, request, datalist, shuffle=True):
        # Training images
        train_d = datalist

        # Validation images
        data_dir = "/home/adp20local/Documents/Datasets/monailabel_datasets/multilabel_abdomen/NIFTI_REORIENTED/val"
        val_images = sorted(glob.glob(os.path.join(data_dir, "imgs", "*.nii.gz")))
        val_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

        return train_d, val_d

    def train_handlers(self, output_dir, events_dir, evaluator, local_rank=0):
        handlers = super().train_handlers(output_dir, events_dir, evaluator, local_rank)
        if self.debug_mode and local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=events_dir))
        return handlers
