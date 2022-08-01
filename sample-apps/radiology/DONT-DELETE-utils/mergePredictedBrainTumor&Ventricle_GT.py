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

import glob
import os

import monai
import numpy as np
import torch
from monai.data import DataLoader, list_data_collate, write_nifti
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, MapTransform, Resized
from monai.utils import GridSampleMode, GridSamplePadMode


class MergeToSingleFiled(MapTransform):
    """
    Merge labels to one channel
    """

    def __call__(self, data):
        d = dict(data)
        factor = 10.0
        # multiply ventricles by factor
        d["GT"] = d["GT"][0] * factor
        print("Size of GT ventricles: ")
        print(d["GT"].shape)
        print("Size of predicted tumors: ")
        print(d["tumors"][0].shape)
        d["label"] = np.add(d["GT"], d["tumors"][0]).astype(np.float32)
        d["label"][d["label"] >= factor] = factor
        d["label"][d["label"] == 3.0] = 4.0
        d["label"][d["label"] == factor] = 5.0
        print(np.unique(d["label"]))
        return d


data_tumors = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/predicted-brain-tumor-validation/monailabel/labels/original/"
data_ventricles_GT = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/ventricles-second-batch/monailabel/labels/final_nrrd/"
output_folder = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/predicted-brain-tumor-validation/mergePredictedTumor&ventriclesGT/"


set_transforms = Compose(
    [
        LoadImaged(keys=("GT", "tumors")),
        EnsureChannelFirstd(keys=("GT", "tumors")),
        Resized(keys="GT", spatial_size=(240, 240, 155), mode="nearest"),
        MergeToSingleFiled(keys="label"),
    ]
)

files_ventricles = glob.glob(os.path.join(data_ventricles_GT, "*"))
train_d = []
for ven_path in files_ventricles:
    dirname, file = os.path.split(ven_path)
    train_d.append({"GT": ven_path, "tumors": data_tumors + file[:-5] + ".nii.gz"})

print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)


def save_nifti(data, meta_data, path, filename):

    spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None
    original_affine = meta_data.get("original_affine", None)[0, ...]
    affine = meta_data.get("affine", None)[0, ...]

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # change data shape to be (channel, h, w, d)
    while len(data.shape) < 4:
        data = np.expand_dims(data, -1)

    # change data to "channel last" format and write to nifti format file
    data = np.moveaxis(np.asarray(data), 0, -1)

    resample = True
    mode = GridSampleMode.BILINEAR
    padding_mode = GridSamplePadMode.BORDER
    align_corners = False
    dtype = np.float64
    output_dtype = np.float32

    write_nifti(
        data,
        file_name=path + filename,
        affine=affine,
        target_affine=original_affine,
        resample=resample,
        output_spatial_shape=spatial_shape,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        dtype=dtype,
        output_dtype=output_dtype,
    )


for idx, img in enumerate(trainLoader):
    dirname, file = os.path.split(img["tumors_meta_dict"]["filename_or_obj"][0])
    print(f"Processing label: {idx}/{len(train_d)} {file}")
    save_nifti(img["label"], img["tumors_meta_dict"], output_folder, file)
