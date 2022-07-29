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

# data_dir = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-third-labels-ventricles/Completed Images-2022-07-28/ALL/"
# output_folder = (
#     "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-third-labels-ventricles/monailabel/labels/final/"
# )
#
# all_files = glob.glob(os.path.join(data_dir, "*/*"))
#
#
# for idx, label_path in enumerate(all_files):
#     dirname, file = os.path.split(label_path)
#     new_filename = dirname.split('/')[-1]
#     print(f"Processing image: {idx}/{len(all_files)}")
#     shutil.copy(label_path, output_folder + new_filename + file[-5:])


source_labels = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-third-labels-ventricles/monailabel/labels/final/"
source_merged_images = "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/test-ventricles/mergedFiles/"
output_folder = (
    "/home/andres/Documents/workspace/Datasets/radiology/BRATS-2021/neuro-atlas-third-labels-ventricles/monailabel/"
)

all_files = glob.glob(os.path.join(source_labels, "*.nrrd"))

for idx, label_path in enumerate(all_files):
    dirname, file = os.path.split(label_path)
    print(f"Processing image: {idx}/{len(all_files)}")
    if os.path.exists(source_merged_images + file[:-5] + ".nii.gz"):
        shutil.copy(source_merged_images + file[:-5] + ".nii.gz", output_folder + file[:-5] + ".nii.gz")
