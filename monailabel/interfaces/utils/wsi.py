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

import copy
import ctypes.util
import logging
import platform
from ctypes import cdll
from math import ceil

from monai.utils import optional_import

logger = logging.getLogger(__name__)


def create_infer_wsi_tasks(request, image):
    if request.get("wsi_tiles"):
        return create_infer_wsi_tasks_from_tiles(request, image)

    tile_size = request.get("tile_size", (2048, 2048))
    tile_size = [int(p) for p in tile_size]

    location = request.get("location", [0, 0])
    size = request.get("size", [0, 0])
    bbox = [[location[0], location[1]], [location[0] + size[0], location[1] + size[1]]]
    bbox = bbox if bbox and sum(bbox[0]) + sum(bbox[1]) > 0 else None

    if platform.system() == "Windows":
        cdll.LoadLibrary(str(ctypes.util.find_library("libopenslide-0.dll")))

    openslide, has_openslide = optional_import("openslide")
    if not has_openslide:
        raise ImportError("Unable to find openslide, please ensure openslide library packages are correctly installed")

    with openslide.OpenSlide(image) as slide:
        w, h = slide.dimensions
    logger.debug(f"Input WSI Image Dimensions: ({w} x {h}); Tile Size: {tile_size}")

    x, y = 0, 0
    if bbox:
        x, y = int(bbox[0][0]), int(bbox[0][1])
        w, h = int(bbox[1][0] - x), int(bbox[1][1] - y)
        logger.debug(f"WSI Region => Location: ({x}, {y}); Dimensions: ({w} x {h})")

    cols = ceil(w / tile_size[0])  # COL
    rows = ceil(h / tile_size[1])  # ROW

    if rows * cols > 1:
        logger.info(f"Total Tiles to infer {rows} x {cols}: {rows * cols}; Dimensions: {w} x {h}")

    infer_tasks = []
    count = 0
    pw, ph = tile_size[0], tile_size[1]
    for row in range(rows):
        for col in range(cols):
            tx = col * pw + x
            ty = row * ph + y

            tw = min(pw, x + w - tx)
            th = min(ph, y + h - ty)

            task = copy.deepcopy(request)
            task.update(
                {
                    "id": count,
                    "image": image,
                    "tile_size": tile_size,
                    "location": (tx, ty),
                    "size": (tw, th),
                }
            )
            infer_tasks.append(task)
            count += 1
    return infer_tasks


def create_infer_wsi_tasks_from_tiles(request, image):
    request = copy.deepcopy(request)
    tiles = request.pop("wsi_tiles")

    infer_tasks = []
    for count, tile in enumerate(tiles):
        tx, ty = tile["location"]
        tw, th = tile["size"]
        task = copy.deepcopy(request)
        task.update(
            {
                "id": count,
                "image": image,
                "tile_size": (tw, th),
                "location": (tx, ty),
                "size": (tw, th),
            }
        )
        infer_tasks.append(task)
    return infer_tasks
