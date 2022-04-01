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

import copy
import logging
from math import ceil

from monai.utils import optional_import

openslide, has_openslide = optional_import("openslide")

logger = logging.getLogger(__name__)


def create_infer_wsi_tasks(request, image):
    tile_size = request.get("tile_size", (2048, 2048))
    tile_size = [int(p) for p in tile_size]

    # TODO:: Auto-Detect based on WSI dimensions instead of 3000
    min_poly_area = request.get("min_poly_area", 3000)

    location = request.get("location", [0, 0])
    size = request.get("size", [0, 0])
    bbox = [[location[0], location[1]], [location[0] + size[0], location[1] + size[1]]]
    bbox = bbox if bbox and sum(bbox[0]) + sum(bbox[1]) > 0 else None
    level = request.get("level", 0)

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
                    "min_poly_area": min_poly_area,
                    "coords": (row, col, tx, ty, tw, th),
                    "location": (tx, ty),
                    "level": level,
                    "size": (tw, th),
                }
            )
            infer_tasks.append(task)
            count += 1
    return infer_tasks
