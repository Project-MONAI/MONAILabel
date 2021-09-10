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
import random
import shutil
import subprocess
from collections import OrderedDict

import numpy as np
from monai.transforms import LoadImage
from tqdm import tqdm

from monailabel.interfaces.exception import MONAILabelError, MONAILabelException

logger = logging.getLogger(__name__)


class ExperimentPlanner(object):
    def __init__(self, datastore):

        self.plans = OrderedDict()
        self.datastore = datastore

        logger.info(f"Available GPU memory: {list(self._get_gpu_memory_map().values())} in MB")
        self._get_img_info()

    def _get_gpu_memory_map(self):
        """Get the current gpu usage.
        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        logger.info("Using nvidia-smi command")
        if shutil.which("nvidia-smi") is None:
            logger.info("nvidia-smi command didn't work! - Using default image size [128, 128, 64]")
            return {0: 4300}

        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], encoding="utf-8"
        )  # --query-gpu=memory.used

        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    def _get_img_info(self):
        loader = LoadImage(reader="nibabelreader")
        spacings = []
        img_sizes = []
        logger.info("Reading datastore metadata for heuristic planner ...")
        if len(self.datastore.list_images()) == 0:
            raise MONAILabelException(
                MONAILabelError.APP_INIT_ERROR,
                "Empty folder!",
            )

        # Sampling 20 images from the datastore
        datastore_check = (
            self.datastore.list_images()
            if len(self.datastore.list_images()) < 20
            else random.sample(self.datastore.list_images(), 20)
        )
        for n in tqdm(datastore_check):
            _, mtdt = loader(self.datastore.get_image_uri(n))
            # Check if images have more than one modality
            if mtdt["pixdim"][4] > 0:
                logger.info(f"Image {mtdt['filename_or_obj'].split('/')[-1]} has more than one modality ...")
            spacings.append(mtdt["pixdim"][1:4])
            img_sizes.append(mtdt["spatial_shape"])
        spacings = np.array(spacings)
        img_sizes = np.array(img_sizes)

        self.target_spacing = np.mean(spacings, 0)
        self.target_img_size = np.mean(img_sizes, 0, np.int64)

    def get_target_img_size(self):
        # This should return an image according to the free gpu memory available
        # Equation obtained from curve fitting using table:
        # https://tinyurl.com/tableGPUMemory
        gpu_mem = self._get_gpu_memory_map()[0]
        # Get a number in base 2 close to the mean depth
        depth_base_2 = int(2 ** np.ceil(np.log2(self.target_img_size[2])))
        # Get the maximum width according available GPU memory
        # This equation roughly estimates the image size that fits in the available GPU memory using DynUNet
        width = (gpu_mem - 2000) / (0.5 * depth_base_2)
        width_base_2 = int(2 ** np.round(np.log2(width)))
        if width_base_2 < np.maximum(self.target_img_size[0], self.target_img_size[1]):
            return [width_base_2, width_base_2, depth_base_2]
        else:
            return [self.target_img_size[0], self.target_img_size[1], depth_base_2]

    def get_target_spacing(self):
        return np.around(self.target_spacing)
