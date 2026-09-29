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

"""DICOM JSON and native frames from locally retained, validated source files."""

from pathlib import Path
from typing import Any

import pydicom
from pydicom.uid import ExplicitVRLittleEndian


def metadata(path: Path) -> dict[str, Any]:
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    result: dict[str, Any] = dataset.to_json_dict()
    # Frames are decoded locally and served in explicit little-endian form.
    result["00020010"] = {"vr": "UI", "Value": [str(ExplicitVRLittleEndian)]}
    return result


def frame(path: Path) -> bytes:
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    return pixels.astype(pixels.dtype.newbyteorder("<"), copy=False).tobytes()
