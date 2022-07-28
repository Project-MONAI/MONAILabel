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
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import DynUNet, SwinUNETR

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.activelearning.tta import TTA
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.tasks.scoring.tta import TTAScoring
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class DeepEdit(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.epistemic_enabled = None
        self.epistemic_samples = None
        self.tta_enabled = None
        self.tta_samples = None

        # Multilabel
        self.labels = {
            "Left-Cerebral-White-Matter": 2,
            "4th-Ventricle": 15,
            "Brain-Stem": 16,
            "Left-Hippocampus": 17,
            "Left-Amygdala": 18,
            "CSF": 24,
            "Left-Accumbens-area": 26,
            "Left-VentralDC": 28,
            "Right-Cerebral-White-Matter": 41,
            "Right-Lateral-Ventricle": 43,
            "Right-Inf-Lat-Vent": 44,
            "Left-Lateral-Ventricle": 4,
            "Right-Cerebellum-White-Matter": 46,
            "Right-Cerebellum-Cortex": 47,
            "Right-Thalamus": 49,
            "Right-Caudate": 50,
            "Right-Putamen": 51,
            "Right-Pallidum": 52,
            "Right-Hippocampus": 53,
            "Right-Amygdala": 54,
            "Right-Accumbens-area": 58,
            "Right-VentralDC": 60,
            "Left-Inf-Lat-Vent": 5,
            "ctx-lh-bankssts": 1001,
            "ctx-lh-caudalanteriorcingulate": 1002,
            "ctx-lh-caudalmiddlefrontal": 1003,
            "ctx-lh-cuneus": 1005,
            "ctx-lh-entorhinal": 1006,
            "ctx-lh-fusiform": 1007,
            "ctx-lh-inferiorparietal": 1008,
            "ctx-lh-inferiortemporal": 1009,
            "ctx-lh-isthmuscingulate": 1010,
            "ctx-lh-lateraloccipital": 1011,
            "Left-Cerebellum-White-Matter": 7,
            "ctx-lh-lateralorbitofrontal": 1012,
            "ctx-lh-lingual": 1013,
            "ctx-lh-medialorbitofrontal": 1014,
            "ctx-lh-middletemporal": 1015,
            "ctx-lh-parahippocampal": 1016,
            "ctx-lh-paracentral": 1017,
            "ctx-lh-parsopercularis": 1018,
            "ctx-lh-parsorbitalis": 1019,
            "ctx-lh-parstriangularis": 1020,
            "ctx-lh-pericalcarine": 1021,
            "Left-Cerebellum-Cortex": 8,
            "ctx-lh-postcentral": 1022,
            "ctx-lh-posteriorcingulate": 1023,
            "ctx-lh-precentral": 1024,
            "ctx-lh-precuneus": 1025,
            "ctx-lh-rostralanteriorcingulate": 1026,
            "ctx-lh-rostralmiddlefrontal": 1027,
            "ctx-lh-superiorfrontal": 1028,
            "ctx-lh-superiorparietal": 1029,
            "ctx-lh-superiortemporal": 1030,
            "ctx-lh-supramarginal": 1031,
            "Left-Thalamus": 10,
            "ctx-lh-frontalpole": 1032,
            "ctx-lh-temporalpole": 1033,
            "ctx-lh-transversetemporal": 1034,
            "ctx-lh-insula": 1035,
            "ctx-rh-bankssts": 2001,
            "ctx-rh-caudalanteriorcingulate": 2002,
            "ctx-rh-caudalmiddlefrontal": 2003,
            "ctx-rh-cuneus": 2005,
            "ctx-rh-entorhinal": 2006,
            "ctx-rh-fusiform": 2007,
            "Left-Caudate": 11,
            "ctx-rh-inferiorparietal": 2008,
            "ctx-rh-inferiortemporal": 2009,
            "ctx-rh-isthmuscingulate": 2010,
            "ctx-rh-lateraloccipital": 2011,
            "ctx-rh-lateralorbitofrontal": 2012,
            "ctx-rh-lingual": 2013,
            "ctx-rh-medialorbitofrontal": 2014,
            "ctx-rh-middletemporal": 2015,
            "ctx-rh-parahippocampal": 2016,
            "ctx-rh-paracentral": 2017,
            "Left-Putamen": 12,
            "ctx-rh-parsopercularis": 2018,
            "ctx-rh-parsorbitalis": 2019,
            "ctx-rh-parstriangularis": 2020,
            "ctx-rh-pericalcarine": 2021,
            "ctx-rh-postcentral": 2022,
            "ctx-rh-posteriorcingulate": 2023,
            "ctx-rh-precentral": 2024,
            "ctx-rh-precuneus": 2025,
            "ctx-rh-rostralanteriorcingulate": 2026,
            "ctx-rh-rostralmiddlefrontal": 2027,
            "Left-Pallidum": 13,
            "ctx-rh-superiorfrontal": 2028,
            "ctx-rh-superiorparietal": 2029,
            "ctx-rh-superiortemporal": 2030,
            "ctx-rh-supramarginal": 2031,
            "ctx-rh-frontalpole": 2032,
            "ctx-rh-temporalpole": 2033,
            "ctx-rh-transversetemporal": 2034,
            "ctx-rh-insula": 2035,
            "3rd-Ventricle": 14,
            "background": 0,
        }

        # Single label
        # self.labels = {
        #     "spleen": 1,
        #     "background": 0,
        # }

        # Number of input channels - 4 for BRATS and 1 for spleen
        self.number_intensity_ch = 4

        network = self.conf.get("network", "dynunet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}_{network}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/deepedit_{network}_all_labels_brats.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        self.spatial_size = (128, 128, 128)  # train input size

        # Network
        self.network = (
            SwinUNETR(
                img_size=self.spatial_size,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
            )
            if network == "swinunetr"
            else DynUNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
            )
        )

        self.network_with_dropout = (
            SwinUNETR(
                img_size=self.spatial_size,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.2,
                use_checkpoint=True,
            )
            if network == "swinunetr"
            else DynUNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
                dropout=0.2,
            )
        )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

        self.tta_enabled = strtobool(conf.get("tta_enabled", "false"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
            ),
            f"{self.name}_seg": lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.SEGMENTATION,
            ),
        }

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network", "dynunet"))
        task: TrainTask = lib.trainers.DeepEdit(
            model_dir=output_dir,
            network=self.network,
            load_path=self.path[0],
            publish_path=self.path[1],
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            number_intensity_ch=self.number_intensity_ch,
            config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
            labels=self.labels,
            debug_mode=False,
            find_unused_parameters=True,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        if self.tta_enabled:
            strategies[f"{self.name}_tta"] = TTA()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=self.network_with_dropout,
                transforms=lib.infers.DeepEdit(
                    type=InferType.DEEPEDIT,
                    path=self.path,
                    network=self.network,
                    labels=self.labels,
                    preload=strtobool(self.conf.get("preload", "false")),
                    spatial_size=self.spatial_size,
                ).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        if self.tta_enabled:
            methods[f"{self.name}_tta"] = TTAScoring(
                model=self.path,
                network=self.network,
                deepedit=True,
                num_samples=self.tta_samples,
                spatial_size=self.spatial_size,
                spacing=self.target_spacing,
            )
        return methods
