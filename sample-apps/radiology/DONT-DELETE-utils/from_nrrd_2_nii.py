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
import shutil

import monai
from monai.data import DataLoader, list_data_collate
from monai.transforms import Compose, LoadImaged, SaveImaged

data_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/ventricles-second-batch/monailabel/labels/final_nrrd/"
output_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/ventricles-second-batch/monailabel/labels/final/"


set_transforms = Compose(
    [
        LoadImaged(keys="label", reader="NrrdReader"),
        SaveImaged(keys="label", output_postfix="", output_dir=output_dir, separate_folder=False),
    ]
)

train_folders = glob.glob(os.path.join(data_dir, "*"))
train_d = []
for image_path in train_folders:
    train_d.append({"label": image_path})

print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)


for idx, img in enumerate(trainLoader):
    print(f"Converting file: {idx}/{len(train_d)}")
