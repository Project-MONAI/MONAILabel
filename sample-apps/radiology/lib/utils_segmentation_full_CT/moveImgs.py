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

data_dir = "/media/andres/disk-workspace/temp-TotalSegmentatorDataset/TotalSegmentatorDataset/Totalsegmentator_dataset/"
output_folder_imgs = "/media/andres/disk-workspace/temp-TotalSegmentatorDataset/monailabel/"

all_folders = glob.glob(os.path.join(data_dir, "*/"))
for idx, image_path in enumerate(all_folders):
    img_name = image_path.split("/")[-2]
    # Copying image
    print(f"Moving image: {img_name} - {idx+1}/{len(all_folders)}")
    shutil.move(os.path.join(image_path, "ct.nii.gz"), os.path.join(output_folder_imgs, img_name + ".nii.gz"))
