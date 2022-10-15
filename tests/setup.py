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

import os
import shutil

from monai.apps import download_url, extractall

TEST_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_DATA = os.path.join(TEST_DIR, "data")


def run_main():
    downloaded_dataset_file = os.path.join(TEST_DIR, "downloads", "dataset.zip")
    dataset_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/test_dataset.zip"
    if not os.path.exists(downloaded_dataset_file):
        download_url(url=dataset_url, filepath=downloaded_dataset_file)
    if not os.path.exists(os.path.join(TEST_DATA, "dataset")):
        extractall(filepath=downloaded_dataset_file, output_dir=TEST_DATA)

    downloaded_pathology_file = os.path.join(TEST_DIR, "downloads", "JP2K-33003-1.svs")
    pathology_url = "https://demo.kitware.com/histomicstk/api/v1/item/5d5c07509114c049342b66f8/download"
    if not os.path.exists(downloaded_pathology_file):
        download_url(url=pathology_url, filepath=downloaded_pathology_file)
    if not os.path.exists(os.path.join(TEST_DATA, "pathology")):
        os.makedirs(os.path.join(TEST_DATA, "pathology"))
        shutil.copy(downloaded_pathology_file, os.path.join(TEST_DATA, "pathology"))

    downloaded_endoscopy_file = os.path.join(TEST_DIR, "downloads", "endoscopy_frames.zip")
    endoscopy_url = "https://drive.google.com/uc?export=download&id=115fS_RZxOXMFb3wgepS2aJA2XYpXjHOU"
    if not os.path.exists(downloaded_endoscopy_file):
        download_url(url=endoscopy_url, filepath=downloaded_endoscopy_file)
    if not os.path.exists(os.path.join(TEST_DATA, "endoscopy")):
        os.makedirs(os.path.join(TEST_DATA, "endoscopy"))
        extractall(filepath=downloaded_endoscopy_file, output_dir=os.path.join(TEST_DATA, "endoscopy"))


if __name__ == "__main__":
    run_main()
