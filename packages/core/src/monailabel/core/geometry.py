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

"""Lossless image-plane transforms; no resampling or intensity changes."""

import numpy as np
from numpy.typing import NDArray

from monailabel.core.models import ImageRegion, PlaneOrientation


def region_slice(region: ImageRegion) -> tuple[slice, slice]:
    """Select a region's bounding rectangle on an H×W source grid."""
    return slice(region.y, region.y + region.height), slice(region.x, region.x + region.width)


def region_pixels(region: ImageRegion) -> NDArray[np.bool_]:
    """Rasterize a validated selection footprint in crop coordinates."""
    shape = (region.height, region.width)
    if region.runs is None:
        return np.ones(shape, dtype=bool)
    pixels = np.zeros(region.height * region.width, dtype=bool)
    for start, stop in region.runs:
        pixels[start:stop] = True
    return pixels.reshape(shape)


def orient_plane[Scalar: np.generic](
    array: NDArray[Scalar], orientation: PlaneOrientation
) -> NDArray[Scalar]:
    if orientation.transpose:
        array = np.swapaxes(array, 0, 1)
    if orientation.flip_rows:
        array = np.flip(array, axis=0)
    if orientation.flip_columns:
        array = np.flip(array, axis=1)
    return array


def restore_plane[Scalar: np.generic](
    array: NDArray[Scalar], orientation: PlaneOrientation
) -> NDArray[Scalar]:
    if orientation.flip_columns:
        array = np.flip(array, axis=1)
    if orientation.flip_rows:
        array = np.flip(array, axis=0)
    if orientation.transpose:
        array = np.swapaxes(array, 0, 1)
    return array
