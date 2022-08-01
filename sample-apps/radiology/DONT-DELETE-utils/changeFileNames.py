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

data_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/brats-ns8/brats-ns8/"
output_folder = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/NeuroAtlas-Labels/brats-ns8/monailabel/labels/final/"

all_files = glob.glob(os.path.join(data_dir, "*.nrrd"))


for idx, img_path in enumerate(all_files):
    fname = img_path.split("/")[-1]
    print(f"Processing image: {idx}/{len(all_files)}")
    # time.sleep(1)
    shutil.copy(img_path, output_folder + fname[:15] + ".nrrd")
