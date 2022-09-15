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

os.environ["FSLDIR"] = "/media/andres/disk-workspace/fsl/"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

source_labels = "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeuroAtlas-Labels/DrTures/all-images/co-registered-volumes/DrTure/merged/"
output_folder = "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeuroAtlas-Labels/DrTures/all-images/co-registered-volumes/DrTure/stripped/"

all_files = glob.glob(os.path.join(source_labels, "*"))

fsl_path = "/media/andres/disk-workspace/fsl/bin/bet"

for idx, label_path in enumerate(all_files):
    dirname, file = os.path.split(label_path)
    name = file.split(".")[0]
    print("Stripping image " + name)
    os.system(fsl_path + " " + source_labels + name + " " + output_folder + name + " -F -f 0.4 -g 0")

print("COMPLETE")
