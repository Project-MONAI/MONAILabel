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

import json
import logging
import pathlib
import shutil
from typing import Any, Callable, Dict, Sequence, Union

import nibabel as nib
import numpy as np
import pkg_resources
import torch
from lib.transforms.transforms import AddEmptySignalChannels, AddGuidanceSignal, NormalizeLabelsInDatasetd
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import Activationsd, AsDiscreted, Compose, EnsureTyped, LoadImaged, Spacingd, SqueezeDimd

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.tasks.infer.basic_infer import BasicInferTask, CallBackTypes

monai_version = pkg_resources.get_distribution("monai").version
if not pkg_resources.parse_version(monai_version) >= pkg_resources.parse_version("1.3.0"):
    raise UserWarning("This code needs at least MONAI 1.3.0")

import os
from pathlib import Path

from monai.transforms import (
    CenterSpatialCropd,
    EnsureChannelFirstd,
    Identityd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SignalFillEmptyd,
)
from monai.utils import set_determinism

# else:
#    from sw_fastedit.utils.helper_transforms import SignalFillEmptyd


logger = logging.getLogger(__name__)


class SWFastEdit(BasicInferTask):

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        label_names=None,
        dimension=3,
        target_spacing=(2.03642011, 2.03642011, 3.0),
        description="",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.label_names = label_names
        self.target_spacing = target_spacing

        set_determinism(42)
        self.model_state_dict = "net"
        self.load_strict = True
        self._amp = True
        # Either no crop with None or crop like (128,128,128), sliding window does not need this parameter unless
        # too much memory is used for the stitching of the output windows
        self.val_crop_size = None

        # Inferer parameters
        # Increase the overlap for up to 1% more Dice, however the time and memory consumption increase a lot!
        self.sw_overlap = 0.25
        # Should be the same ROI size as it was trained on
        self.sw_roi_size = (128, 128, 128)

        # Reduce this if you run into OOMs
        self.train_sw_batch_size = 8
        # Reduce this if you run into OOMs
        self.val_sw_batch_size = 16

    def __call__(self, request, callbacks=None):
        if callbacks is None:
            callbacks = {}
        callbacks[CallBackTypes.POST_TRANSFORMS] = post_callback

        return super().__call__(request, callbacks)

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        # print("#########################################")
        # data['label_dict'] = self.label_names
        data["label_names"] = self.label_names

        # Make sure the click keys already exist
        for label in self.label_names:
            if not label in data:
                data[label] = []
        # data['click_path'] = self.click_path

        cpu_device = torch.device("cpu")
        device = data.get("device") if data else None
        loglevel = logging.DEBUG
        input_keys = "image"

        t = []
        t_val_1 = [
            # InitLoggerd(loglevel=loglevel, no_log=True, log_dir=None),
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            # ScaleIntensityRangePercentilesd(
            #     keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            # ),
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True),
            SignalFillEmptyd(keys=input_keys),
        ]
        t.extend(t_val_1)
        # self.add_cache_transform(t, data)
        t_val_2 = [
            AddEmptySignalChannels(keys=input_keys, device=device),
            AddGuidanceSignal(
                keys=input_keys,
                sigma=1,
                disks=True,
                device=device,
            ),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=self.target_spacing),
            (
                CenterSpatialCropd(keys=input_keys, roi_size=self.val_crop_size)
                if self.val_crop_size is not None
                else Identityd(keys=input_keys, allow_missing_keys=True)
            ),
            EnsureTyped(keys=input_keys, device=device),
        ]
        t.extend(t_val_2)
        return t

    def inferer(self, data=None) -> Inferer:
        sw_params = {
            "roi_size": self.sw_roi_size,
            "mode": "gaussian",
            "cache_roi_weight_map": False,
            "overlap": self.sw_overlap,
        }
        eval_inferer = SlidingWindowInferer(sw_batch_size=self.val_sw_batch_size, **sw_params)
        return eval_inferer

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        device = data.get("device") if data else None
        return [
            EnsureTyped(keys="pred", device=device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            EnsureTyped(keys="pred", device="cpu" if data else None, dtype=torch.uint8),
        ]


def post_callback(data):
    """
    Saves clicks in the same folder where the created labels are stored.
    Can also help debugging by providing a way of saving nifti files.
    """
    image_name = Path(os.path.basename(data["image_path"]))
    true_image_name = image_name.name.removesuffix("".join(image_name.suffixes))
    image_folder = Path(data["image_path"]).parent

    labels_folder = os.path.join(image_folder, "labels", "final")
    if not os.path.exists(labels_folder):
        print(f"##### Creating {labels_folder}")
        pathlib.Path(labels_folder).mkdir(parents=True)

    # Save the clicks
    clicks_per_label = {}
    for key in data["label_names"].keys():
        clicks_per_label[key] = data[key]
        assert isinstance(data[key], list)

    click_file_path = os.path.join(labels_folder, f"{true_image_name}_clicks.json")
    logger.info(f"Now dumping dict: {clicks_per_label} to file {click_file_path} ...")
    with open(click_file_path, "w") as clicks_file:
        json.dump(clicks_per_label, clicks_file)

    # Save debug NIFTI, not fully working since the inverse transform of the image is not avaible
    if False:
        logger.info("SAVING NIFTI")
        inputs = data["image"]
        pred = data["pred"]
        logger.info(f"inputs.shape is {inputs.shape}")
        logger.info(f"sum of fgg is {torch.sum(inputs[1])}")
        logger.info(f"sum of bgg is {torch.sum(inputs[2])}")
        logger.info(f"Image path is {data['image_path']}, copying file")
        shutil.copyfile(data["image_path"], f"{path}/im.nii.gz")
        # save_nifti(f"{path}/im", inputs[0].cpu().detach().numpy())
        save_nifti(f"{path}/guidance_fgg", inputs[1].cpu().detach().numpy())
        save_nifti(f"{path}/guidance_bgg", inputs[2].cpu().detach().numpy())
        logger.info(f"pred.shape is {pred.shape}")
        save_nifti(f"{path}/pred", pred.cpu().detach().numpy())
    return data


def save_nifti(name, im):
    """ONLY FOR DEBUGGING"""
    affine = np.eye(4)
    affine[0][0] = -1
    ni_img = nib.Nifti1Image(im, affine=affine)
    ni_img.header.get_xyzt_units()
    ni_img.to_filename(f"{name}.nii.gz")
