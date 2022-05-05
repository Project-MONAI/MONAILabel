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
import logging
import os

import cv2
import numpy as np
import skimage
import torch
from ignite.metrics import Accuracy
from lib.handlers import TensorBoardImageHandler
from lib.transforms import FilterImaged
from lib.utils import split_dataset
from monai.apps.deepgrow.transforms import AddGuidanceSignald, AddRandomGuidanced, FindDiscrepancyRegionsd
from monai.handlers import from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    RandRotate90d,
    ScaleIntensityRangeD,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
    Transform,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class NuClick(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        tile_size=(256, 256),
        roi_size=(128, 128),
        description="Pathology NuClick Segmentation",
        **kwargs,
    ):
        self._network = network
        self.labels = labels
        self.tile_size = tile_size
        self.roi_size = roi_size
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(sigmoid=True, squared_pred=True)

    def pre_process(self, request, datastore: Datastore):
        self.cleanup(request)

        cache_dir = os.path.join(self.get_cache_dir(request), "train_ds")
        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        return split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=self.labels,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )

    def get_click_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            ToNumpyd(keys=("image", "label", "pred")),
            FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanced(guidance="guidance", discrepancy="discrepancy", probability="probability"),
            AddGuidanceSignald(image="image", guidance="guidance", number_intensity_ch=3),
            ToTensord(keys=("image", "label")),
        ]

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=5),
            AsChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            ExtractPatchd(),
            EnsureTyped(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]

    def train_key_metric(self, context: Context):
        return {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_key_metric(self, context: Context):
        return {"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))}

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir, batch_limit=4))
        return handlers


class ExtractPatchd(Transform):
    def __init__(self, image="image", label="label", centroid="centroid"):
        self.image = image
        self.label = label
        self.centroid = centroid

        self.patch_size = 128
        self.perturb = "distance"
        self.drop_rate = 0.5
        self.jitter_range = 3

    def __call__(self, data):
        d = dict(data)

        img = d[self.image]
        mask = d[self.label][0]
        y, x = d[self.centroid]
        m, n = mask.shape[:2]

        x_start = int(max(x - self.patch_size / 2, 0))
        y_start = int(max(y - self.patch_size / 2, 0))
        x_end = x_start + self.patch_size
        y_end = y_start + self.patch_size
        if x_end > n:
            x_end = n
            x_start = n - self.patch_size
        if y_end > m:
            y_end = m
            y_start = m - self.patch_size

        mask_val = mask[y, x]

        mask_patch = mask[y_start:y_end, x_start:x_end]
        img_patch = img[:, y_start:y_end, x_start:x_end]

        mask_patch_in = (mask_patch == mask_val).astype(np.uint8)
        others_patch_in = (1 - mask_patch_in) * mask_patch
        others_patch_in = self.mask_relabeling(others_patch_in, size_limit=5).astype(np.uint8)

        pad_size = (self.patch_size, self.patch_size)
        img_patch = self.pad_to_shape(img_patch, pad_size, False)
        mask_patch_in = self.pad_to_shape(mask_patch_in, pad_size, True)
        others_patch_in = self.pad_to_shape(others_patch_in, pad_size, True)

        # create the guiding signals
        signal_gen = PointGuidingSignal(mask_patch_in, others_patch_in, perturb=self.perturb)
        inc_signal = signal_gen.inclusion_map()
        exc_signal = signal_gen.exclusion_map(random_drop=self.drop_rate, random_jitter=self.jitter_range)

        image_patch = np.concatenate((img_patch, inc_signal[np.newaxis, ...], exc_signal[np.newaxis, ...]), axis=0)
        d[self.image] = image_patch
        d[self.label] = mask_patch_in[np.newaxis]
        return d

    @staticmethod
    def pad_to_shape(img, shape, is_mask):
        img_shape = img.shape[-2:]
        shape_diff = np.array(shape) - np.array(img_shape)
        if is_mask:
            img_padded = np.pad(img, [(0, shape_diff[0]), (0, shape_diff[1])], mode="constant", constant_values=0)
        else:
            img_padded = np.pad(
                img, [(0, 0), (0, shape_diff[0]), (0, shape_diff[1])], mode="constant", constant_values=0
            )
        return img_padded

    @staticmethod
    def mask_relabeling(mask, size_limit=5):
        out_mask = np.zeros_like(mask, dtype=np.uint16)
        unique_labels = np.unique(mask)
        if unique_labels[0] == 0:
            unique_labels = np.delete(unique_labels, 0)

        i = 1
        for l in unique_labels:
            m = skimage.measure.label(mask == l, connectivity=1)
            stats = skimage.measure.regionprops(m)
            for stat in stats:
                if stat.area > size_limit:
                    out_mask[stat.coords[:, 0], stat.coords[:, 1]] = i
                    i += 1
        return out_mask


def adaptive_distance_thresholding(mask):
    """Refining the input mask using adaptive distance thresholding.

    Distance map of the input mask is generated and the an adaptive
    (random) threshold based on the distance map is calculated to
    generate a new mask from distance map based on it.

    Inputs:
        mask (::np.ndarray::): Should be a 2D binary numpy array (uint8)
    Outputs:
        new_mask (::np.ndarray::): the refined mask
        dist (::np.ndarray::): the distance map
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
    tempMean = np.mean(dist[dist > 0])
    tempStd = np.std(dist[dist > 0])
    tempTol = tempStd / 2
    low_thresh = np.max([tempMean - tempTol, 0])
    high_thresh = np.min([tempMean + tempTol, np.max(dist) - tempTol])
    if low_thresh >= high_thresh:
        thresh = tempMean
    else:
        thresh = np.random.uniform(low_thresh, high_thresh)
    new_mask = dist > thresh
    if np.all(new_mask == np.zeros_like(new_mask)):
        new_mask = dist > tempMean
    return new_mask, dist


class GuidingSignal:
    """A generic class for defining guiding signal generators.

    This class include some special methods that inclusion and exclusion guiding signals
    for different application can be created based on.
    """

    def __init__(self, mask: np.ndarray, others: np.ndarray, kernel_size: int = 0) -> None:
        self.mask = self.mask_validator(mask > 0.5)
        self.kernel_size = kernel_size
        if kernel_size:
            self.current_mask = self.mask_preprocess(self.mask, kernel_size=self.kernel_size)
        else:
            self.current_mask = self.mask_validator(mask > 0.5)
        self.others = others

    @staticmethod
    def mask_preprocess(mask, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if np.all(mask == np.zeros_like(mask)):
            logging.warning(
                f"The kernel_size (radius) of {kernel_size} may be too high, consider checking "
                "the intermediate output for the sanity of generated masks."
            )
        return mask

    @staticmethod
    def mask_validator(mask):
        """Validate the input mask be np.uint8 and 2D"""
        assert len(mask.shape) == 2, "Mask must be a 2D array (NxM)"
        if not issubclass(type(mask[0, 0]), np.integer):
            mask = np.uint8(mask)
        return mask

    def inclusion_map(self):
        """A function to generate inclusion gioding signal"""
        raise NotImplementedError

    def exclusion_map(self):
        """A function to generate exclusion gioding signal"""
        raise NotImplementedError


class PointGuidingSignal(GuidingSignal):
    def __init__(self, mask: np.ndarray, others: np.ndarray, perturb: str = "None", **kwargs) -> None:
        super().__init__(mask, others, **kwargs)
        if perturb.lower() not in {"none", "distance", "inside"}:
            raise ValueError(
                f'Invalid running perturb type of: {perturb}. Perturn type should be `"None"`, `"inside"`, or `"distance"`.'
            )
        self.perturb = perturb.lower()

    def inclusion_map(self):
        if self.perturb is None:  # if there is no purturbation
            indices = np.argwhere(self.current_mask == 1)  #
            centroid = np.mean(indices, axis=0)
            pointMask = np.zeros_like(self.current_mask)
            pointMask[int(centroid[0]), int(centroid[1]), 0] = 1
            return pointMask, self.current_mask
        elif self.perturb == "distance" and np.any(self.current_mask > 0):
            new_mask, _ = adaptive_distance_thresholding(self.current_mask)
        else:  # if self.perturb=='inside':
            new_mask = self.current_mask.copy()

        # Creating the point map
        pointMask = np.zeros_like(self.current_mask)
        indices = np.argwhere(new_mask == 1)
        if len(indices) > 0:
            rndIdx = np.random.randint(0, len(indices))
            rndX = indices[rndIdx, 1]
            rndY = indices[rndIdx, 0]
            pointMask[rndY, rndX] = 1

        return pointMask

    def exclusion_map(self, random_drop=0.0, random_jitter=0):
        _, _, _, centroids = cv2.connectedComponentsWithStats(self.others, 4, cv2.CV_32S)

        centroids = centroids[1:, :]  # removing the first centroid, it's background
        if random_jitter:
            centroids = self.jitterClicks(self.current_mask.shape, centroids, jitter_range=random_jitter)
        if random_drop:  # randomly dropping some of the points
            drop_prob = np.random.uniform(0, random_drop)
            num_select = int((1 - drop_prob) * centroids.shape[0])
            select_indices = np.random.choice(centroids.shape[0], size=num_select, replace=False)
            centroids = centroids[select_indices, :]
        centroids = np.int64(np.floor(centroids))

        # create the point map
        pointMask = np.zeros_like(self.others)
        pointMask[centroids[:, 1], centroids[:, 0]] = 1

        return pointMask

    @staticmethod
    def jitterClicks(shape, centroids, jitter_range=3):
        """Randomly jitter the centroid points
        Points should be an array in (x, y) format while shape is (H, W) of the point map
        """
        centroids += np.random.uniform(low=-jitter_range, high=jitter_range, size=centroids.shape)
        centroids[:, 0] = np.clip(centroids[:, 0], 0, shape[1] - 1)
        centroids[:, 1] = np.clip(centroids[:, 1], 0, shape[0] - 1)
        return centroids
