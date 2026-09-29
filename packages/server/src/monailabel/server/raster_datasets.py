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

"""Published 2D dataset layouts and source groups, without resizing reference masks."""

import io
from collections import defaultdict
from itertools import zip_longest
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.server.data import MAX_IMAGE_PIXELS
from monailabel.server.dataset_downloads import Archive, Source

RASTER_FORMATS = {"tnbc", "kvasir-instrument"}


def image_names(source: Source, archive: Archive) -> list[str]:
    if source.format == "tnbc":
        groups: dict[str, list[str]] = defaultdict(list)
        for name in sorted(archive.names):
            path = PurePosixPath(name)
            if path.parent.name.startswith("Slide_") and path.suffix == ".png":
                groups[path.parent.name].append(name)
        # A small demonstration includes independent patients, not just patches
        # from the first slide. The ordering is stable across repeated imports.
        return [name for row in zip_longest(*groups.values()) for name in row if name]
    return sorted(
        name
        for name in archive.names
        if PurePosixPath(name).parent.name == "images" and name.endswith(".jpg")
    )


def source_group(source: Source, name: str) -> str:
    path = PurePosixPath(name)
    if source.format == "tnbc":
        return "tnbc:" + path.parent.name.removeprefix("Slide_")
    if source.format == "kvasir-instrument":
        # The release has frame identifiers but no patient/procedure mapping.
        # Its published frame split cannot establish independent source groups.
        return "kvasir-instrument:unknown-procedures"
    if source.format == "msd":
        return f"msd:{source.dataset_id or source.id}:{path.name}"
    if source.format == "totalsegmentator-mr":
        return f"totalsegmentator-mr:{path.parent.name}"
    return f"totalsegmentator:{path.parent.name}"


def binary_mask(source: Source, archive: Archive, name: str, shape: tuple[int, ...]) -> np.ndarray:
    path = PurePosixPath(name)
    if source.format == "tnbc":
        mask_name = path.parent.parent / path.parent.name.replace("Slide_", "GT_") / path.name
    else:
        mask_name = path.parent.parent / "masks" / (path.stem + ".png")
    try:
        with Image.open(io.BytesIO(archive.read(str(mask_name)))) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise DomainError("Reference mask exceeds the local pixel limit.")
            values = np.asarray(image).copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise DomainError("Could not read the dataset reference mask.") from error
    if values.ndim == 3:
        if values.shape[-1] != 3 or not np.all(values == values[..., :1]):
            raise DomainError("Expected a grayscale binary reference mask.")
        values = values[..., 0]
    if values.shape != shape:
        raise DomainError("Reference mask dimensions differ from its image.")
    if not set(np.unique(values)) <= {0, 255}:
        raise DomainError("Expected background 0 and foreground 255 in this dataset mask.")
    return np.asarray(values == 255, dtype=np.uint8)


def png_mask(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(mask).save(stream, format="PNG")
    return stream.getvalue()
