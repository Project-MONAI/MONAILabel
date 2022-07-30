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
from monai.transforms import Compose, LoadImaged, MapTransform
from monai.utils import GridSampleMode, GridSamplePadMode


class MergeToSingleFiled(MapTransform):
    """
    Merge labels to one channel
    """

    def __call__(self, data):
        d = dict(data)
        factor = 10.0
        # Take only ventricles from prediction
        d['ventricles'][d['ventricles'] != 4.0] = 0
        d['ventricles'][d['ventricles'] == 4.0] = 50.0
        # multiply GT by factor
        d['GT'] = d['GT'] * factor
        d['label'] = np.add(d['ventricles'], d['GT']).astype(np.float32)
        d['label'][d['label'] == (factor + 50.0)] = 1.0
        d['label'][d['label'] == factor] = 1.0
        d['label'][d['label'] == (2 * factor) + 50.0] = 2.0
        d['label'][d['label'] == (2 * factor)] = 2.0
        d['label'][d['label'] == (4 * factor) + 50.0] = 4.0
        d['label'][d['label'] == (4 * factor)] = 4.0
        d['label'][d['label'] == 50.0] = 5.0
        print(np.unique(d['label']))
        return d


data_ventricles = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/predicted_ventricles/monailabel/labels/original/"
data_tumor_GT = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/to_train_tumors_only/monailabel/labels/final/"
output_folder = (
    "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/predicted_ventricles/mergeVentricles&GTTumors/"
)


set_transforms = Compose(
    [
        LoadImaged(keys=("ventricles", "GT")),
        MergeToSingleFiled(keys="label"),
    ]
)

files_ventricle = glob.glob(os.path.join(data_ventricles, "*"))
train_d = []
for ventricle_path in files_ventricle:
    dirname, file = os.path.split(ventricle_path)
    train_d.append({"ventricles": ventricle_path, "GT": data_tumor_GT + file})

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
    dirname, file = os.path.split(img["GT_meta_dict"]["filename_or_obj"][0])
    print("Processing label: ", file + ".nii.gz")
    save_nifti(img["label"], img["GT_meta_dict"], output_folder, file)
