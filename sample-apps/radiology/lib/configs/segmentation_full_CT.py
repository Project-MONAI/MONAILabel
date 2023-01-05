# Copyright (c) MONAI Consortium
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
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import SegResNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationFullCT(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "spleen": 1,
            "kidney_right": 2,
            "kidney_left": 3,
            "gallbladder": 4,
            "liver": 5,
            "stomach": 6,
            "aorta": 7,
            "inferior_vena_cava": 8,
            "portal_vein_and_splenic_vein": 9,
            "pancreas": 10,
            "adrenal_gland_right": 11,
            "adrenal_gland_left": 12,
            "lung_upper_lobe_left": 13,
            "lung_lower_lobe_left": 14,
            "lung_upper_lobe_right": 15,
            "lung_middle_lobe_right": 16,
            "lung_lower_lobe_right": 17,
            "vertebrae_L5": 18,
            "vertebrae_L4": 19,
            "vertebrae_L3": 20,
            "vertebrae_L2": 21,
            "vertebrae_L1": 22,
            "vertebrae_T12": 23,
            "vertebrae_T11": 24,
            "vertebrae_T10": 25,
            "vertebrae_T9": 26,
            "vertebrae_T8": 27,
            "vertebrae_T7": 28,
            "vertebrae_T6": 29,
            "vertebrae_T5": 30,
            "vertebrae_T4": 31,
            "vertebrae_T3": 32,
            "vertebrae_T2": 33,
            "vertebrae_T1": 34,
            "vertebrae_C7": 35,
            "vertebrae_C6": 36,
            "vertebrae_C5": 37,
            "vertebrae_C4": 38,
            "vertebrae_C3": 39,
            "vertebrae_C2": 40,
            "vertebrae_C1": 41,
            "esophagus": 42,
            "trachea": 43,
            "heart_myocardium": 44,
            "heart_atrium_left": 45,
            "heart_ventricle_left": 46,
            "heart_atrium_right": 47,
            "heart_ventricle_right": 48,
            "pulmonary_artery": 49,
            "brain": 50,
            "iliac_artery_left": 51,
            "iliac_artery_right": 52,
            "iliac_vena_left": 53,
            "iliac_vena_right": 54,
            "small_bowel": 55,
            "duodenum": 56,
            "colon": 57,
            "rib_left_1": 58,
            "rib_left_2": 59,
            "rib_left_3": 60,
            "rib_left_4": 61,
            "rib_left_5": 62,
            "rib_left_6": 63,
            "rib_left_7": 64,
            "rib_left_8": 65,
            "rib_left_9": 66,
            "rib_left_10": 67,
            "rib_left_11": 68,
            "rib_left_12": 69,
            "rib_right_1": 70,
            "rib_right_2": 71,
            "rib_right_3": 72,
            "rib_right_4": 73,
            "rib_right_5": 74,
            "rib_right_6": 75,
            "rib_right_7": 76,
            "rib_right_8": 77,
            "rib_right_9": 78,
            "rib_right_10": 79,
            "rib_right_11": 80,
            "rib_right_12": 81,
            "humerus_left": 82,
            "humerus_right": 83,
            "scapula_left": 84,
            "scapula_right": 85,
            "clavicula_left": 86,
            "clavicula_right": 87,
            "femur_left": 88,
            "femur_right": 89,
            "hip_left": 90,
            "hip_right": 91,
            "sacrum": 92,
            "face": 93,
            "gluteus_maximus_left": 94,
            "gluteus_maximus_right": 95,
            "gluteus_medius_left": 96,
            "gluteus_medius_right": 97,
            "gluteus_minimus_left": 98,
            "gluteus_minimus_right": 99,
            "autochthon_left": 100,
            "autochthon_right": 101,
            "iliopsoas_left": 102,
            "iliopsoas_right": 103,
            "urinary_bladder": 104,
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_segmentation_segresnet_full_CT_15mm.pt"
            # url = f"{url}/radiology_segmentation_segresnet_full_CT_2mm.pt" # 2mm pretrained model
            download_file(url, self.path[0])

        self.target_spacing = (1.5, 1.5, 1.5)  # target space for image
        # Setting ROI size - This is for the image padding
        self.roi_size = (96, 96, 96)

        # Network
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(self.labels) + 1,  # labels plus background,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationFullCT(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationFullCT(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train Full Segmentation Model",
            labels=self.labels,
        )
        return task
