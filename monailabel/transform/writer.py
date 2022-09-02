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
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import itk
import nrrd
import numpy as np
import torch
from monai.data import MetaTensor, write_nifti

from monailabel.utils.others.generic import file_ext
from monailabel.utils.others.pathology import create_asap_annotations_xml, create_dsa_annotations_json

logger = logging.getLogger(__name__)


# TODO:: Move to MONAI ??
def write_itk(image_np, output_file, affine, dtype, compress):
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.numpy()
    if len(image_np.shape) >= 2:
        image_np = image_np.transpose().copy()
    if dtype:
        image_np = image_np.astype(dtype)

    result_image = itk.image_from_array(image_np)
    logger.debug(f"ITK Image size: {itk.size(result_image)}")

    # https://github.com/RSIP-Vision/medio/blob/master/medio/metadata/affine.py#L108-L121
    if affine is not None:
        convert_aff_mat = np.diag([-1, -1, 1, 1])
        if affine.shape[0] == 3:
            convert_aff_mat = np.diag([-1, -1, 1])
        affine = convert_aff_mat @ affine

        dim = affine.shape[0] - 1
        _origin_key = (slice(-1), -1)
        _m_key = (slice(-1), slice(-1))

        origin = affine[_origin_key]
        spacing = np.linalg.norm(affine[_m_key] @ np.eye(dim), axis=0)
        direction = affine[_m_key] @ np.diag(1 / spacing)

        logger.debug(f"Affine: {affine}")
        logger.debug(f"Origin: {origin}")
        logger.debug(f"Spacing: {spacing}")
        logger.debug(f"Direction: {direction}")

        result_image.SetDirection(itk.matrix_from_array(direction))
        result_image.SetSpacing(spacing)
        result_image.SetOrigin(origin)

    itk.imwrite(result_image, output_file, compress)


def write_seg_nrrd(
    image_np: np.ndarray,
    output_file: str,
    dtype: type,
    affine: np.ndarray,
    labels: List[str],
    color_map: Optional[Dict[str, List[float]]] = None,
    index_order: str = "C",
    space: str = "left-posterior-superior",
) -> None:
    """Write multi-channel seg.nrrd file.

    Args:
        image_np: Image as numpy ndarray
        output_file: Output file path that the seg.nrrd file should be saved to
        dtype: numpy type e.g. float32
        affine: Affine matrix
        labels: Labels of image segment which will be written to the nrrd header
        color_map: Mapping from segment_name(str) to it's color e.g. {'heart': [255/255, 244/255, 209/255]}
        index_order: Either 'C' or 'F' (see nrrd.write() documentation)

    Raises:
        ValueError: In case affine is not provided
        ValueError: In case labels are not provided
    """
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.numpy()
    image_np = image_np.transpose().copy()
    if dtype:
        image_np = image_np.astype(dtype)

    if not isinstance(labels, Iterable):
        raise ValueError("Labels have to be defined, e.g. as a list")

    header: Dict[str, Any] = {}
    for i, segment_name in enumerate(labels):
        header.update(
            {
                f"Segment{i}_ID": segment_name,
                f"Segment{i}_Name": segment_name,
            }
        )
        if color_map is not None:
            header[f"Segment{i}_Color"] = " ".join(list(map(str, color_map[segment_name])))

    if affine is None:
        raise ValueError("Affine matrix has to be defined")

    kinds = ["list", "domain", "domain", "domain"]

    convert_aff_mat = np.diag([-1, -1, 1, 1])
    affine = convert_aff_mat @ affine

    _origin_key = (slice(-1), -1)
    origin = affine[_origin_key]

    space_directions = np.array(
        [
            [np.nan, np.nan, np.nan],
            affine[0, :3],
            affine[1, :3],
            affine[2, :3],
        ]
    )

    header.update(
        {
            "kinds": kinds,
            "space directions": space_directions,
            "space origin": origin,
            "space": space,
        }
    )
    nrrd.write(
        output_file,
        image_np,
        header=header,
        index_order=index_order,
    )


class Writer:
    def __init__(
        self,
        label="pred",
        json=None,
        ref_image=None,
        key_extension="result_extension",
        key_dtype="result_dtype",
        key_compress="result_compress",
        key_write_to_file="result_write_to_file",
        meta_key_postfix="meta_dict",
        nibabel=False,
    ):
        self.label = label
        self.json = json
        self.ref_image = ref_image if ref_image else label

        # User can specify through params
        self.key_extension = key_extension
        self.key_dtype = key_dtype
        self.key_compress = key_compress
        self.key_write_to_file = key_write_to_file
        self.meta_key_postfix = meta_key_postfix
        self.nibabel = nibabel

    def __call__(self, data) -> Tuple[Any, Any]:
        logger.setLevel(data.get("logging", "INFO").upper())

        path = data.get("image_path")
        ext = file_ext(path) if path else None
        dtype = data.get(self.key_dtype, None)
        compress = data.get(self.key_compress, False)
        write_to_file = data.get(self.key_write_to_file, True)

        ext = data.get(self.key_extension) if data.get(self.key_extension) else ext
        write_to_file = write_to_file if ext else False
        logger.info(f"Result ext: {ext}; write_to_file: {write_to_file}; dtype: {dtype}")

        if isinstance(data[self.label], MetaTensor):
            image_np = data[self.label].array
        else:
            image_np = data[self.label]

        # Always using Restored as the last transform before writing
        meta_dict = data.get(f"{self.ref_image}_{self.meta_key_postfix}")
        affine = meta_dict.get("affine") if meta_dict else None
        if affine is None and isinstance(data[self.ref_image], MetaTensor):
            affine = data[self.ref_image].affine

        logger.debug(f"Image: {image_np.shape}; Data Image: {data[self.label].shape}")

        output_file = None
        output_json = data.get(self.json, {})
        if write_to_file:
            output_file = tempfile.NamedTemporaryFile(suffix=ext).name
            logger.debug(f"Saving Image to: {output_file}")

            if self.is_multichannel_image(image_np):
                if ext != ".seg.nrrd":
                    logger.warning(
                        f"Using extension '{ext}' with multi-channel 4D label will probably fail"
                        + "Consider to use extension '.seg.nrrd'"
                    )
                labels = data.get("labels")
                color_map = data.get("color_map")
                logger.debug("Using write_seg_nrrd...")
                write_seg_nrrd(image_np, output_file, dtype, affine, labels, color_map)
            # Issue with slicer:: https://discourse.itk.org/t/saving-non-orthogonal-volume-in-nifti-format/2760/22
            elif self.nibabel and ext and ext.lower() in [".nii", ".nii.gz"]:
                logger.debug("Using MONAI write_nifti...")
                write_nifti(image_np, output_file, affine=affine, output_dtype=dtype)
            else:
                write_itk(image_np, output_file, affine if len(image_np.shape) > 2 else None, dtype, compress)
        else:
            output_file = image_np

        return output_file, output_json

    def is_multichannel_image(self, image_np: np.ndarray) -> bool:
        """Check if the provided image contains multiple channels

        Args:
            image_np : Expected shape (channels, width, height, batch)

        Returns:
            bool: If this is a multi-channel image or not
        """
        return len(image_np.shape) == 4 and image_np.shape[0] > 1


class ClassificationWriter:
    def __init__(self, label="pred", label_names=None):
        self.label = label
        self.label_names = label_names

    def __call__(self, data):
        logger.info(data[self.label].array)

        result = []
        for idx, score in enumerate(data[self.label]):
            name = f"label_{idx}"
            name = self.label_names.get(idx) if self.label_names else name
            if name:
                result.append({"idx": idx, "label": name, "score": float(score)})

        return None, {"prediction": result}


class PolygonWriter:
    def __init__(
        self,
        label="pred",
        json="result",
        key_write_to_file="result_write_to_file",
        key_annotations="annotations",
        key_label_colors="label_colors",
        key_output_format="output",
    ):
        self.label = label
        self.json = json
        self.key_write_to_file = key_write_to_file
        self.key_annotations = key_annotations
        self.key_label_colors = key_label_colors
        self.key_output_format = key_output_format
        self.format = format

    def __call__(self, data):
        loglevel = data.get("logging", "INFO").upper()
        logger.setLevel(loglevel)

        output = data.get(self.key_output_format, "dsa")
        logger.info(f"+++ Output Type: {output}")

        output_json = data.get(self.json, {})
        write_to_file = data.get(self.key_write_to_file, True)
        if not write_to_file:
            return None, output_json

        res_json = {
            "name": f"MONAILabel Annotations - {data.get('model')}",
            "description": data.get("description"),
            "model": data.get("model"),
            "location": data.get("location"),
            "size": data.get("size"),
            "annotations": [output_json],
            "latencies": data.get("latencies"),
        }

        output_file = None
        if output == "asap":
            logger.info("+++ Generating ASAP XML Annotation")
            output_file, _ = create_asap_annotations_xml(res_json, loglevel=loglevel)
        elif output == "dsa":
            logger.info("+++ Generating DSA JSON Annotation")
            output_file, _ = create_dsa_annotations_json(res_json, loglevel=loglevel)
        else:
            logger.info("+++ Return Default JSON Annotation")

        return output_file, res_json
