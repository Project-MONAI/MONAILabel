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
import math
import pathlib

import numpy as np
import openslide
import torch
from monai.apps.pathology.transforms import GenerateInstanceType
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import BoundingRect, MapTransform, Pad, Transform
from monai.utils import HoVerNetBranch, PostFix, convert_to_numpy, ensure_tuple
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_holes, remove_small_objects

from monailabel.utils.others.label_colors import get_color

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __init__(self, keys: KeysCollection, mode="RGB", dtype=np.uint8, padding=True):
        super().__init__(keys)
        self.mode = mode
        self.dtype = dtype
        self.padding = padding

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], str):
                continue  # Support direct image in np (pass only transform)

            name = d[key]
            ext = pathlib.Path(name).suffix
            if ext == ".npy":
                d[key] = np.load(d[key])
                continue

            location = d.get("location", (0, 0))
            level = d.get("level", 0)
            size = d.get("size", None)

            # Model input size
            tile_size = d.get("tile_size", size)

            if not ext or ext in (
                ".bif",
                ".mrxs",
                ".ndpi",
                ".scn",
                ".svs",
                ".svslide",
                ".tif",
                ".tiff",
                ".vms",
                ".vmu",
            ):
                slide = openslide.OpenSlide(name)
                size = size if size else slide.dimensions
                img = slide.read_region(location, level, size)
            else:
                img = Image.open(d[key])
                d["location"] = [0, 0]
                d["size"] = [0, 0]

            img = img.convert(self.mode) if self.mode else img
            image_np = np.array(img, dtype=self.dtype)
            image_np = np.moveaxis(image_np, 0, 1)

            meta_dict_key = f"{key}_{PostFix.meta()}"
            meta_dict = d.get(meta_dict_key)
            if meta_dict is None:
                d[meta_dict_key] = dict()
                meta_dict = d.get(meta_dict_key)

            meta_dict["spatial_shape"] = np.asarray(image_np.shape[:-1])
            meta_dict["original_channel_dim"] = -1
            meta_dict["original_affine"] = None  # type: ignore
            logger.debug(f"Image shape: {image_np.shape} vs size: {size} vs tile_size: {tile_size}")

            if self.padding and tile_size and (image_np.shape[0] != tile_size[0] or image_np.shape[1] != tile_size[1]):
                image_np = self.pad_to_shape(image_np, tile_size)
            d[key] = MetaTensor(image_np, meta=meta_dict)
        return d

    @staticmethod
    def pad_to_shape(img, shape):
        img_shape = img.shape[:-1]
        s_diff = np.array(shape) - np.array(img_shape)
        diff = [(0, s_diff[0]), (0, s_diff[1]), (0, 0)]
        return np.pad(
            img,
            diff,
            mode="constant",
            constant_values=0,
        )


class PostFilterLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, min_size=64, min_hole=64):
        super().__init__(keys)
        self.min_size = min_size
        self.min_hole = min_hole

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = convert_to_numpy(d[key]) if isinstance(d[key], torch.Tensor) else d[key]
            label = label.astype(np.uint8)
            if self.min_hole:
                label = remove_small_holes(label, area_threshold=self.min_hole)
            label = binary_fill_holes(label).astype(np.uint8)
            if self.min_size:
                label = remove_small_objects(label, min_size=self.min_size)

            d[key] = np.where(label > 0, d[key], 0)
        return d


class ConvertInteractiveClickSignals(MapTransform):
    """
    ConvertInteractiveClickSignals converts interactive annotation information (e.g. from DSA) into a format expected
    by NuClick. Typically, it will take point annotations from data["annotations"][<source_annotation_key>], convert
    it to 2d points, and place it in data[<target_data_key>].
    """

    def __init__(
        self, source_annotation_keys: KeysCollection, target_data_keys: KeysCollection, allow_missing_keys: bool = False
    ):
        super().__init__(target_data_keys, allow_missing_keys)
        self.source_annotation_keys = ensure_tuple(source_annotation_keys)
        self.target_data_keys = ensure_tuple(target_data_keys)

    def __call__(self, data):
        data = dict(data)
        annotations = data.get("annotations", {})
        annotations = {} if annotations is None else annotations
        for source_annotation_key, target_data_key in zip(self.source_annotation_keys, self.target_data_keys):
            if source_annotation_key in annotations:
                points = annotations.get(source_annotation_key)["points"]
                print(f"points={points}")
                points = [coords[0:2] for coords in points]
                data[target_data_key] = points
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"source_annotation_key={source_annotation_key} not found in annotations.keys()={annotations.keys()}"
                )
        return data


class ContoursFromHovernetInstanceInfod(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        result="result",
        result_output_key="annotation",
        key_label_colors="label_colors",
        labels=None,
        colormap=None,
    ):
        super().__init__(keys)

        self.result = result
        self.result_output_key = result_output_key
        self.key_label_colors = key_label_colors
        self.colormap = colormap

        labels = labels if labels else dict()
        labels = [labels] if isinstance(labels, str) else labels
        if not isinstance(labels, dict):
            labels = {v: k + 1 for k, v in enumerate(labels)}

        labels = {v: k for k, v in labels.items()}
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        location = d.get("location", [0, 0])
        size = d.get("size", [0, 0])
        color_map = d.get(self.key_label_colors) if self.colormap is None else self.colormap

        elements = []
        label_names = set()
        for key in self.keys:
            labels = {}
            for instance in d[key].values():
                label_idx = instance["type"]
                if not labels.get(label_idx):
                    labels[label_idx] = []
                labels[label_idx].append(instance)
            logger.info(f"Total Unique Masks (excluding background): {list(labels.keys())}")

            for label_idx, instances in labels.items():
                label_name = self.labels.get(label_idx, label_idx)
                label_names.add(label_name)

                polygons = []
                contours = [instance["contour"] for instance in instances]
                for contour in contours:
                    if len(contour) < 3:
                        continue

                    contour[:, 0] += location[0]  # X
                    contour[:, 1] += location[1]  # Y

                    coords = contour.astype(int).tolist()
                    polygons.append(coords)

                if len(polygons):
                    logger.info(f"+++++ {label_idx} => Total Polygons Found: {len(polygons)}")
                    elements.append({"label": label_name, "contours": polygons})

        if elements:
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result][self.result_output_key] = {
                "location": location,
                "size": size,
                "elements": elements,
                "labels": {n: get_color(n, color_map) for n in label_names},
            }
            logger.debug(f"+++++ ALL => Total Annotation Elements Found: {len(elements)}")
        return d


class ToHoverNetPatchesd(Transform):
    def __init__(self, image="image", input_size=(270, 270), output_size=(80, 80)):
        self.image = image
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, data):
        d = dict(data)

        img = d[self.image]
        w = img.shape[-2]
        h = img.shape[-1]

        x = self.output_size[0]
        y = self.output_size[1]

        # debug = data.get("debug", False)

        win_size = self.input_size
        msk_size = step_size = self.output_size

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        last_w, _ = get_last_steps(w, msk_size[0], step_size[0])
        last_h, _ = get_last_steps(h, msk_size[1], step_size[1])

        padl = (win_size[0] - step_size[0]) // 2
        padt = (win_size[1] - step_size[1]) // 2
        padr = last_w + win_size[0] - w
        padb = last_h + win_size[1] - h

        padding = Pad()
        img = padding(img, to_pad=[(0, 0), (padl, padr), (padt, padb)], mode="reflect")

        patches = []
        for i in range(math.ceil(w / x)):
            for j in range(math.ceil(h / y)):
                x1 = i * self.output_size[0]
                y1 = j * self.output_size[1]
                x2 = x1 + self.input_size[0]
                y2 = y1 + self.input_size[1]

                p = img[:, x1:x2, y1:y2]
                patches.append(p)

                # if debug:
                #     p = p[:3] * 255
                #     p = torch.moveaxis(p, 0, -1).type(torch.uint8)
                #     im = Image.fromarray(p.array, mode="RGB")
                #     im.save(f"/localhome/sachi/Dataset/Pathology/dummy/patches/img/{i}x{j}.png")

        d[self.image] = torch.stack(patches, dim=0)
        d["image_spatial_size"] = (w, h)
        return d


class FromHoverNetPatchesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        image_spatial_size="image_spatial_size",
        output_size=(80, 80),
    ):
        super().__init__(keys, allow_missing_keys)

        self.image_spatial_size = image_spatial_size
        self.output_size = output_size

    def __call__(self, data):
        d = dict(data)

        img = d[self.image_spatial_size]
        w = img[-2]
        h = img[-1]

        x = self.output_size[0]
        y = self.output_size[1]
        # debug = data.get("debug", False)

        for key in self.key_iterator(d):
            patches = d[key]
            c = patches[0].shape[0]

            count = 0
            pred = torch.zeros((c, w, h), dtype=patches.dtype)
            for i in range(math.ceil(w / x)):
                for j in range(math.ceil(h / y)):
                    x1 = i * self.output_size[0]
                    y1 = j * self.output_size[1]
                    x2 = min(w, x1 + self.output_size[0])
                    y2 = min(h, y1 + self.output_size[1])

                    p = patches[count]
                    pred[:, x1:x2, y1:y2] = p[:, 0 : (x2 - x1), 0 : (y2 - y1)]
                    count += 1

                    # if debug and key == "nucleus_prediction":
                    #     p = torch.softmax(p, dim=0)
                    #     p = torch.argmax(p, dim=0, keepdim=True)
                    #     p[p > 0] = 255
                    #     p = p[0].type(torch.uint8)
                    #     im = Image.fromarray(p.array if isinstance(p, MetaTensor) else p.cpu().detach().numpy())
                    #     im.save(f"/localhome/sachi/Dataset/Pathology/dummy/patches/lab/{i}x{j}.png")

            d[key] = pred
        return d


class RenameKeyd(Transform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        d = dict(data)
        d[self.target_key] = d.pop(self.source_key)
        return d


class HoverNetPostProcessWS(Transform):
    def __init__(
        self,
        output_key="pred",
        mask_key="mask",
        labels=None,
    ):
        self.output_key = output_key
        self.mask_key = mask_key
        self.labels = {v: k for k, v in labels.items()}

    def __call__(self, data):
        d = dict(data)

        seg_pred = d[self.mask_key]
        type_pred = d[HoVerNetBranch.NC]

        result = torch.zeros(seg_pred.shape, dtype=torch.uint8)
        inst_id_list = sorted(torch.unique(seg_pred))[1:]  # exclude background
        for inst_id in inst_id_list:
            inst_map = torch.where(seg_pred == inst_id, 1, 0)
            inst_bbox = BoundingRect()(inst_map)

            inst_type, type_prob = GenerateInstanceType()(
                bbox=inst_bbox,
                type_pred=type_pred,
                seg_pred=seg_pred,
                instance_id=inst_id,
            )
            result = torch.where(seg_pred == inst_id, inst_type, result)

        logger.info(f"Total Labels Types: {torch.unique(result)}")
        d[self.output_key] = result
        return d
