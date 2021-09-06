# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import shutil
import subprocess
from collections import OrderedDict

import numpy as np
from monai.transforms import LoadImage
from tqdm import tqdm

from monailabel.interfaces import MONAILabelError, MONAILabelException

logger = logging.getLogger(__name__)


class ExperimentPlanner(object):
    def __init__(self, datastore):

        self.plans = OrderedDict()
        self.datastore = datastore
        self.get_img_info()

    def get_gpu_memory_map(self):
        """Get the current gpu usage.
        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        logger.info("Using nvidia-smi command")
        if shutil.which("nvidia-smi") is None:
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                "nvidia-smi command doesn't work!",
            )
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], encoding="utf-8"
        )

        # --query-gpu=memory.used

        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

        return gpu_memory_map

    def get_img_info(self):
        loader = LoadImage(reader="ITKReader")
        spacings = []
        img_sizes = []
        logger.info("Reading datastore metadata for heuristic planner ...")
        if len(self.datastore.list_images()) == 0:
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                "Empty folder!",
            )
        for n in tqdm(self.datastore.list_images()):
            _, mtdt = loader(self.datastore.get_image_uri(n))
            spacings.append(mtdt["spacing"])
            img_sizes.append(mtdt["spatial_shape"])
        spacings = np.array(spacings)
        img_sizes = np.array(img_sizes)

        self.target_spacing = np.mean(spacings, 0)
        self.target_img_size = np.mean(img_sizes, 0, np.int64)
        # Changing from DHW to HDW order
        self.target_img_size = np.array([self.target_img_size[1], self.target_img_size[2], self.target_img_size[0]])

    def get_target_img_size(self):
        # This should return an image according to the free gpu memory available
        # These values are for DynUNetV1. In Megabytes
        memory_use = [3000, 4100, 4300, 5900, 7700, 9000, 9300, 12100, 17700]
        sizes = {
            "3000": [64, 64, 32],
            "4100": [256, 256, 16],
            "4300": [128, 128, 64],
            "5900": [256, 256, 32],
            "7700": [192, 192, 96],
            "9000": [256, 256, 64],
            "9300": [192, 192, 128],
            "12100": [256, 256, 96],
            "17700": [256, 256, 128],
        }
        idx = np.abs(np.array(memory_use) - self.get_gpu_memory_map()[0]).argmin()
        img_size_gpu = sizes[str(memory_use[idx])]
        if img_size_gpu[0] > self.target_img_size[0]:
            return self.target_img_size
        else:
            return sizes[str(memory_use[idx])]

    def get_target_spacing(self):
        return np.around(self.target_spacing)
