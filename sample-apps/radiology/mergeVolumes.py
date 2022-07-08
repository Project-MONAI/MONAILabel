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
import time

import shutil

import os
import glob
import monai
from monai.data import DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    SaveImaged,
)

data_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/test-ventricles/individual-files/"
output_folder = '/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/test-ventricles/mergedFiles/'


set_transforms = Compose(
                        [
                            LoadImaged(keys="image"),
                            SaveImaged(keys="image", output_postfix='', output_dir=output_folder, separate_folder=False),
                        ]
                        )

train_folders = glob.glob(os.path.join(data_dir, "*/"))
train_d = []
for image_path in train_folders:
    file_name = image_path.split('/')[-2]
    path_imgs = []
    for mod in ['flair.nii.gz', 't1.nii.gz', 't1ce.nii.gz', 't2.nii.gz']:
        path_imgs.append(image_path + file_name + '_' + mod)
    train_d.append({"image": path_imgs})

print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)


for idx, img in enumerate(trainLoader):
    dirname, file = os.path.split(img['image_meta_dict']['filename_or_obj'][0])
    fname = dirname.split('/')[-1]
    print('Processing image: ', fname + '.nii.gz')
    time.sleep(2)
    shutil.move(output_folder + file, output_folder + fname + '.nii.gz')