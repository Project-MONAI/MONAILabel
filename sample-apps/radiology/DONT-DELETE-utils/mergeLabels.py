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
        for key in self.keys:
            # multiply ventricles by factor
            if len(np.unique(d[key][0, ...])) == 2:
                d[key][0, ...] = d[key][0, ...] * factor
            else:
                d[key][1, ...] = d[key][1, ...] * factor
            d[key] = np.add(d[key][0, ...], d[key][1, ...]).astype(np.float32)
            d[key][d[key] >= factor] = factor
            d[key][d[key] == 4.0] = 3.0
            d[key][d[key] == factor] = 4.0
        return d


data_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-first-labels/ventricles/"
output_folder = (
    "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-first-labels/mergedFiles/labels/"
)


set_transforms = Compose(
    [
        LoadImaged(keys="label"),
        MergeToSingleFiled(keys="label"),
    ]
)

train_folders = glob.glob(os.path.join(data_dir, "*/"))
train_d = []
for image_path in train_folders:
    file_name = image_path.split("/")[-2]
    path_labels = []
    all_nifti = glob.glob(os.path.join(image_path + "/*.nii.gz"))
    for mod in all_nifti:
        dirname, file = os.path.split(mod)
        if "_seg" in mod:
            path_labels.append(mod)
        elif "Label" in mod:
            path_labels.append(mod)
        elif "Untitled" in mod:
            path_labels.append(mod)
        # ['seg.nii.gz', 'flair.nii.gz', 't1.nii.gz', 't1ce.nii.gz', 't2.nii.gz']
    train_d.append({"label": path_labels})

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
    dirname, file = os.path.split(img["label_meta_dict"]["filename_or_obj"][0])
    fname = dirname.split("/")[-1]
    print("Processing label: ", fname + ".nii.gz")
    save_nifti(img["label"], img["label_meta_dict"], output_folder, fname + ".nii.gz")
