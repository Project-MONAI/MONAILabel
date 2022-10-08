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

data_dir = "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeurosurgicalAtlas/DrTures/nrrd-nifti-files/co-registered-volumes/all-raw-nrrd/"
output_folder = "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeurosurgicalAtlas/DrTures/nrrd-nifti-files/DrTure-ns98/monailabel-CORRECTED-RESIZED/"


set_transforms = Compose(
    [
        LoadImaged(keys="image"),
        SaveImaged(keys="image", output_postfix="", output_dir=output_folder, separate_folder=False),
    ]
)

folders = glob.glob(os.path.join(data_dir, "*"))
train_d = []
for path in folders:
    patient = path.split("/")[-1]
    path_imgs = []
    for mod in ["-FLAIR", "-T1", "-T1C", "-T2"]:
        path_img = data_dir + patient + "/" + patient + mod + ".nrrd"
        if os.path.exists(path_img):
            # WHEN MISSING VOLUME, DO SOMETHING SMARTER ... SHOULD WE REPEAT LAST MODALITY?
            path_imgs.append(path_img)
        else:
            pass
    if len(path_imgs) == 4:
        train_d.append({"image": path_imgs})
    else:
        pass


print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)


for idx, img in enumerate(trainLoader):
    dirname, file = os.path.split(img["image_meta_dict"]["filename_or_obj"][0])
    fname = dirname.split("/")[-1]
    print("Processing image: ", fname + ".nii.gz")
    # time.sleep(2)
    shutil.move(output_folder + os.path.splitext(file)[0] + ".nii.gz", output_folder + fname + ".nii.gz")
