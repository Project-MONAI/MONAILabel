import subprocess
from collections import OrderedDict

import numpy as np
from monai.transforms import LoadImage


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
        for n in self.datastore.list_images():
            _, mtdt = loader(self.datastore.get_image_uri(n))
            spacings.append(mtdt["spacing"])
            img_sizes.append(mtdt["spatial_shape"])
        spacings = np.array(spacings)
        img_sizes = np.array(img_sizes)

        self.target_spacing = np.mean(spacings, 0)
        self.target_img = np.mean(img_sizes, 0)

    def get_target_img_size(self):
        # This should return an image according to the free gpu memory available
        return (
            np.array(self.target_img * 0.5, dtype=np.int32)
            if self.target_img[0] >= 512
            else np.array(self.target_img * 0.7, dtype=np.int32)
        )

    def get_target_spacing(self):
        return self.target_spacing
