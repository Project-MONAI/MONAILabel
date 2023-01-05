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

from monai.transforms import LoadImage

source_labels = "/media/andres/disk-workspace/temp-TotalSegmentatorDataset/monailabel/labels/final"

all_files = glob.glob(os.path.join(source_labels, "*.nii.gz"))

for idx, img_path in enumerate(all_files):
    d = LoadImage(image_only=True)(img_path)
    dirname, file = os.path.split(img_path)
    print(f"Image: {file} => max intensity {d.array.max()} - min intensity {d.array.min()}")
    # print("Unique values: ", np.unique(d.array))
    # print(f"Labelmap size: {d.array.shape}")
