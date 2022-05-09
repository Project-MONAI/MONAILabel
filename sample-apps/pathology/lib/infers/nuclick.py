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
from typing import Any, Callable, Dict, Sequence

import numpy as np
from lib.transforms import LoadImagePatchd
from monai.config import KeysCollection
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    EnsureTyped,
    MapTransform,
    SqueezeDimd,
    ToNumpyd,
    Transform,
)
from skimage.morphology import disk, reconstruction, remove_small_holes, remove_small_objects

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class NuClick(InferTask):
    """
    This provides Inference Engine for pre-trained NuClick segmentation (UNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(128, 128),
        type=InferType.OTHERS,
        labels=None,
        dimension=2,
        description="A pre-trained NuClick model for interactive cell segmentation for Pathology",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        d["nuclick"] = True
        return d

    def pre_transforms(self, data=None):
        return [
            LoadImagePatchd(keys="image", conversion="RGB", dtype=np.uint8, padding=False),
            AsChannelFirstd(keys="image"),
            AddClickSignalsd(image="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        return super().run_inferer(data, False, device)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            SqueezeDimd(keys="pred", dim=1),
            ToNumpyd(keys=("image", "pred")),
            PostFilterLabeld(keys="pred"),
            FindContoursd(keys="pred", labels=self.labels),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)


class AddClickSignalsd(Transform):
    def __init__(self, image, foreground="foreground"):
        self.image = image
        self.foreground = foreground

    def __call__(self, data):
        d = dict(data)

        location = d.get("location", (0, 0))
        tx, ty = location[0], location[1]

        pos = d.get(self.foreground)
        pos = (np.array(pos) - (tx, ty)).astype(int).tolist() if pos else []

        cx = [xy[0] for xy in pos]
        cy = [xy[1] for xy in pos]

        img = d[self.image].astype(np.uint8)
        img_width = img.shape[-1]
        img_height = img.shape[-2]

        click_map, bounding_boxes = self.get_clickmap_boundingbox(cx, cy, img_height, img_width)
        patches, nuc_points, other_points = self.get_patches_and_signals(
            img, click_map, bounding_boxes, cx, cy, img_height, img_width
        )
        patches = patches / 255

        d["bounding_boxes"] = bounding_boxes
        d["img_width"] = img_width
        d["img_height"] = img_height
        d["nuc_points"] = nuc_points

        d[self.image] = np.concatenate((patches, nuc_points, other_points), axis=1, dtype=np.float32)
        return d

    @staticmethod
    def get_clickmap_boundingbox(cx, cy, m, n, bb=128):
        click_map = np.zeros((m, n), dtype=np.uint8)

        # Removing points out of image dimension (these points may have been clicked unwanted)
        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        click_map[cy, cx] = 1
        bounding_boxes = []
        for i in range(len(cx)):
            x_start = cx[i] - bb // 2
            y_start = cy[i] - bb // 2
            if x_start < 0:
                x_start = 0
            if y_start < 0:
                y_start = 0
            x_end = x_start + bb - 1
            y_end = y_start + bb - 1
            if x_end > n - 1:
                x_end = n - 1
                x_start = x_end - bb + 1
            if y_end > m - 1:
                y_end = m - 1
                y_start = y_end - bb + 1
            bounding_boxes.append([x_start, y_start, x_end, y_end])
        return click_map, bounding_boxes

    @staticmethod
    def get_patches_and_signals(img, click_map, bounding_boxes, cx, cy, m, n, bb=128):
        # total = number of clicks
        total = len(bounding_boxes)
        img = np.array([img])  # img.shape=(1,3,m,n)
        click_map = np.array([click_map])  # clickmap.shape=(1,m,n)
        click_map = click_map[:, np.newaxis, ...]  # clickmap.shape=(1,1,m,n)

        patches = np.ndarray((total, 3, bb, bb), dtype=np.uint8)
        nuc_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)
        other_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)

        # Removing points out of image dimension (these points may have been clicked unwanted)
        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        for i in range(len(bounding_boxes)):
            bounding_box = bounding_boxes[i]
            x_start = bounding_box[0]
            y_start = bounding_box[1]
            x_end = bounding_box[2]
            y_end = bounding_box[3]

            patches[i] = img[0, :, y_start : y_end + 1, x_start : x_end + 1]

            this_click_map = np.zeros((1, 1, m, n), dtype=np.uint8)
            this_click_map[0, 0, cy[i], cx[i]] = 1

            others_click_map = np.uint8((click_map - this_click_map) > 0)

            nuc_points[i] = this_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]
            other_points[i] = others_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]

        # patches: (total, 3, m, n)
        # nuc_points: (total, 1, m, n)
        # other_points: (total, 1, m, n)
        return patches, nuc_points, other_points


class PostFilterLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        nuc_points="nuc_points",
        bounding_boxes="bounding_boxes",
        img_height="img_height",
        img_width="img_width",
        thresh=0.33,
        min_size=10,
        min_hole=30,
        do_reconstruction=False,
    ):
        super().__init__(keys)
        self.nuc_points = nuc_points
        self.bounding_boxes = bounding_boxes
        self.img_height = img_height
        self.img_width = img_width

        self.thresh = thresh
        self.min_size = min_size
        self.min_hole = min_hole
        self.do_reconstruction = do_reconstruction

    def __call__(self, data):
        d = dict(data)

        nuc_points = d[self.nuc_points]
        bounding_boxes = d[self.bounding_boxes]
        img_height = d[self.img_height]
        img_width = d[self.img_width]

        for key in self.keys:
            label = d[key].astype(np.uint8)
            masks = self.post_processing(
                label,
                thresh=self.thresh,
                min_size=self.min_size,
                min_hole=self.min_hole,
                do_reconstruction=self.do_reconstruction,
                nuc_points=nuc_points,
            )

            d[key] = self.gen_instance_map(masks, bounding_boxes, img_height, img_width).astype(np.uint8)
        return d

    @staticmethod
    def post_processing(preds, thresh=0.33, min_size=10, min_hole=30, do_reconstruction=False, nuc_points=None):
        masks = preds > thresh
        masks = remove_small_objects(masks, min_size=min_size)
        masks = remove_small_holes(masks, area_threshold=min_hole)
        if do_reconstruction:
            for i in range(len(masks)):
                this_mask = masks[i]
                this_marker = nuc_points[i, 0, :, :] > 0

                try:
                    this_mask = reconstruction(this_marker, this_mask, footprint=disk(1))
                    masks[i] = np.array([this_mask])
                except:
                    logger.warning("Nuclei reconstruction error #" + str(i))
        return masks  # masks(no.patches, 128, 128)

    @staticmethod
    def gen_instance_map(masks, bounding_boxes, m, n, flatten=True):
        instance_map = np.zeros((m, n), dtype=np.uint16)
        for i in range(len(masks)):
            this_bb = bounding_boxes[i]
            this_mask_pos = np.argwhere(masks[i] > 0)
            this_mask_pos[:, 0] = this_mask_pos[:, 0] + this_bb[1]
            this_mask_pos[:, 1] = this_mask_pos[:, 1] + this_bb[0]
            instance_map[this_mask_pos[:, 0], this_mask_pos[:, 1]] = 1 if flatten else i + 1
        return instance_map
