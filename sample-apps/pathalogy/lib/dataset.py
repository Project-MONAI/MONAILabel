from typing import Callable, Optional, Sequence

from monai.data import Dataset
from PIL import Image


class GridImageDataset(Dataset):
    def __init__(self, data: Sequence, image_size, patch_size, transform: Optional[Callable] = None):
        super().__init__(data, transform)

        if image_size % patch_size != 0:
            raise Exception("Image size / patch size != 0 : {} / {}".format(image_size, patch_size))

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_per_side = self.image_size // self.patch_size
        self.grid_size = self.patch_per_side * self.patch_per_side

    def __getitem__(self, idx):
        image = []
        label = []

        return (image, label)
