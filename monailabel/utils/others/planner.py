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

import numpy as np
from monai.transforms import LoadImage
from tqdm import tqdm

from monailabel.utils.others.generic import gpu_memory_map

logger = logging.getLogger(__name__)


class HeuristicPlanner(object):
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 64), max_samples=10):

        self.target_spacing = target_spacing
        self.spatial_size = spatial_size
        self.max_samples = max_samples
        self.max_pix = None
        self.min_pix = None
        self.mean_pix = None
        self.std_pix = None

    def run(self, datastore):
        logger.info("Reading datastore metadata for heuristic planner...")
        if len(datastore.list_images()) == 0:
            logger.warning("Currently no images are available in datastore for sampling")
            return

        # Sampling max_samples images from the datastore
        datastore_check = (
            datastore.list_images()
            if len(datastore.list_images()) < self.max_samples
            else random.sample(datastore.list_images(), self.max_samples)
        )

        spacings = []
        img_sizes = []
        pix_img_max = []
        pix_img_min = []
        pix_img_mean = []
        pix_img_std = []
        loader = LoadImage()
        for n in tqdm(datastore_check):
            img, mtdt = loader(datastore.get_image_uri(n))

            # Check if images have more than one modality
            if mtdt["pixdim"][4] > 0:
                logger.info(f"Image {mtdt['filename_or_obj'].split('/')[-1]} has more than one modality ...")
            spacings.append(mtdt["pixdim"][1:4])
            img_sizes.append(mtdt["spatial_shape"])

            pix_img_max.append(img.max())
            pix_img_min.append(img.min())
            pix_img_mean.append(img.mean())
            pix_img_std.append(img.std())

        spacings = np.array(spacings)
        img_sizes = np.array(img_sizes)

        logger.info(f"Available GPU memory: {gpu_memory_map()} in MB")

        self.target_spacing = self._get_target_spacing(np.mean(spacings, 0))
        self.spatial_size = self._get_target_img_size(np.mean(img_sizes, 0, np.int64))
        logger.info(f"Spacing: {self.target_spacing}; Spatial Size: {self.spatial_size}")

        # Image stats for intensity normalization
        self.max_pix = np.max(np.array(pix_img_max))
        self.min_pix = np.min(np.array(pix_img_min))
        self.mean_pix = np.mean(np.array(pix_img_mean))
        self.std_pix = np.mean(np.array(pix_img_std))
        logger.info(f"Maximum pixel value: {self.max_pix}; Minimum pixel value: {self.min_pix}")

    @staticmethod
    def _get_target_img_size(target_img_size):
        # This should return an image according to the free gpu memory available
        # Equation obtained from curve fitting using table:
        # https://tinyurl.com/tableGPUMemory
        gpu_mem = gpu_memory_map()[0]
        # Get a number in base 2 close to the mean depth
        depth_base_2 = int(2 ** np.ceil(np.log2(target_img_size[2])))
        # Get the maximum width according available GPU memory
        # This equation roughly estimates the image size that fits in the available GPU memory using DynUNet
        width = (gpu_mem - 2000) / (0.5 * depth_base_2)
        width_base_2 = int(2 ** np.round(np.log2(width)))
        if width_base_2 < np.maximum(target_img_size[0], target_img_size[1]):
            return [width_base_2, width_base_2, depth_base_2]
        else:
            return [target_img_size[0], target_img_size[1], depth_base_2]

    @staticmethod
    def _get_target_spacing(target_spacing):
        return np.around(target_spacing)
