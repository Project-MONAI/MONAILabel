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
