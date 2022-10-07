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
import argparse
import glob
import os

from monai.transforms import LoadImage


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument(
        "-i",
        "--input",
        default="/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeuroAtlas-Labels/DrTures/test-raw-dicom/",
    )
    args = parser.parse_args()

    images = glob.glob(os.path.join(args.input, "*/*/*"))

    # define transforms for image
    load_transform = LoadImage(reader="ITKReader", series_meta=True)

    for x in images:
        print(f"Reading image: {x}")
        try:
            test_img = load_transform(x)
        except:
            print("Error")
            pass


if __name__ == "__main__":
    main()
