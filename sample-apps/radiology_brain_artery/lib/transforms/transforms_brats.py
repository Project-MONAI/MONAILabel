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
import tempfile
from typing import Dict, Iterable, Any, List, Optional, Tuple

import numpy as np
import nrrd
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, LoadImage, Spacing, EnsureChannelFirst, Resize
from monailabel.utils.others.generic import file_ext

logger = logging.getLogger(__name__)


class GetSingleModalityBRATSd(MapTransform):
    """
    Gets one modality

    "0": "FLAIR",
    "1": "T1w",
    "2": "t1gd",
    "3": "T2w"

    """

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                # TRANSFORM IN PROGRESS - SHOULD AFFINE AND ORIGINAL BE CHANGED??

                # Output is only one channel
                # Get T1 Gadolinium. Better to describe brain tumour. FLAIR is better for edema (swelling in the brain)
                d[key] = d[key][..., 2]
                # d['image_meta_dict']["original_affine"] = d['image_meta_dict']["original_affine"][0, ...]
                # d['image_meta_dict']["affine"] = d['image_meta_dict']["affine"][0, ...]
                # d[key] = d[key][None] # Add dimension
            else:
                print("Transform 'GetSingleModalityBRATSd' only works for the image key")

        return d


class BinarySegd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for k in self.key_iterator(data):
            data[k] = (data[k] > 0).to(torch.float32)
        return data

class MaskTumord(MapTransform):
    """
    Mask the tumor and edema labels

    """

    def __call__(self, data):
        d: Dict = dict(data)
        mask_tumor_edema = copy.deepcopy(d["label"])
        mask_tumor_edema[mask_tumor_edema < 15] = 1
        mask_tumor_edema[mask_tumor_edema > 1] = 0
        for key in self.key_iterator(d):
            if len(d[key].shape) == 4:
                for i in range(d[key].shape[-1]):
                    d[key][..., i] = d[key][..., i] * mask_tumor_edema
            else:
                d[key] = d[key] * mask_tumor_edema
        return d


class MergeLabelsd(MapTransform):
    """
    Merge conflicting labels into single one

    """

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            merged_labels = copy.deepcopy(d[key])
            merged_labels[merged_labels == 6] = 6  # Thalamus
            merged_labels[merged_labels == 7] = 6  # Caudate
            merged_labels[merged_labels == 8] = 8  # Putamen
            merged_labels[merged_labels == 9] = 8  # Pallidum
            d[key] = merged_labels
        return d

class LoadOtherModalitiesd(MapTransform):
    """
    Merge all modalities into one

    "0": "FLAIR",
    "1": "T1",
    "2": "T1C",
    "3": "T2"

    """

    def __init__(self, keys: KeysCollection,  target_spacing, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.target_spacing = target_spacing

    def __call__(self, data):
        d: Dict = dict(data)
        input_path = '/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeurosurgicalAtlas/DrTures/nrrd-nifti-files/co-registered-volumes/cases-with-4-modalities/all-four/'
        for key in self.key_iterator(d):
            img = copy.deepcopy(d[key])
            name = img.meta['filename_or_obj'].split('/')[-1].split('.')[0]
            all_mods = np.zeros(
                (4, img.array.shape[-3], img.array.shape[-2], img.array.shape[-1]), dtype=np.float32
            )
            all_mods[0, ...] = img.array[0, ...]
            logging.info(f'Size of the images in transform: {all_mods.shape}')
            ## Cleaner solution?
            # trans = Compose(
            #     [
            #         LoadImage(reader='ITKReader'),
            #         EnsureChannelFirst(),
            #         Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            #     ]
            # )
            loader = LoadImage(reader='ITKReader')
            chan_first = EnsureChannelFirst()
            spacer = Spacing(pixdim=self.target_spacing, mode="bilinear")
            res = Resize(spatial_size=(img.array.shape[-3], img.array.shape[-2], img.array.shape[-1]))
            for idx, mod in enumerate(['-T1.nrrd', '-T1C.nrrd', '-T2.nrrd']):
                aux = loader(input_path + name + mod)
                logging.info(f'Modality: {mod} is being read')
                aux = chan_first(aux[0])
                spaced_img = spacer(aux)
                logging.info(f'Modality: {mod} is being spaced')
                # This solves the issue of images having slightly different resolution
                resized_img = res(spaced_img)
                all_mods[idx+1, ...] = resized_img[0].array
            d['spatial_size'] = all_mods[0, ...]
            d[key].array = all_mods
        return d



class RestoreOriginalIndexingd(MapTransform):
    """
    Restore original indexing
    """

    def __init__(self, keys: KeysCollection,  original_label_indexing, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.original_label_indexing = original_label_indexing

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            label = np.zeros(d[key].shape)
            for idx, (key_label, val_label) in enumerate(self.original_label_indexing.items(), start=1):
                label[d[key] == idx] = val_label
            d[key].array = label
        return d


def write_seg_nrrd_brain(
    image_np: np.ndarray,
    output_file: str,
    dtype: type,
    affine: np.ndarray,
    labels: List[str],
    color_map: Optional[Dict[str, List[float]]] = None,
    index_order: str = "C",
    space: str = "left-posterior-superior",
) -> None:
    """Write seg.nrrd file for Neuro Atlas work

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
    for i, (segment_name, val_label) in enumerate(labels.items()):
        header.update(
            {
                f"Segment{i}_ID": segment_name,
                f"Segment{i}_LabelValue": val_label,
                f"Segment{i}_Layer": '0',
                f"Segment{i}_Name": segment_name,
            }
        )
        if color_map is not None:
            header[f"Segment{i}_Color"] = " ".join(list(map(str, color_map[segment_name])))

    if affine is None:
        raise ValueError("Affine matrix has to be defined")

    kinds = ["domain", "domain", "domain"]

    _origin_key = (slice(-1), -1)
    origin = affine[_origin_key]
    origin = origin * [-1, -1, 1]

    convert_aff_mat = np.diag([-1, -1, 1, 1])
    affine = convert_aff_mat @ affine

    space_directions = np.array(
        [
            affine[0, :3],
            affine[1, :3],
            affine[2, :3],
        ]
    )

    space_directions = np.transpose(space_directions)

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


class NRRDWriterBrain:
    def __init__(
            self,
            label="pred",
            original_label_indexing=None,
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
        self.original_label_indexing = original_label_indexing
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
            if self.original_label_indexing is None:
                raise ValueError("Missing labels")
            else:
                labels = self.original_label_indexing
            color_map = None
            logger.debug("Using write_seg_nrrd...")
            write_seg_nrrd_brain(image_np, output_file, dtype, affine, labels, color_map)
        else:
            output_file = image_np

        return output_file, output_json

class AddUnknownLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, max_labels=None, allow_missing_keys: bool = False):
        """
        Assign unknown label to intensities bigger than 0 - background is anything else

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.max_labels = max_labels

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            unknown_mask = copy.deepcopy(d["image"][..., 0])
            unknown_mask[unknown_mask > unknown_mask.max() * 0.10] = 1
            mask_all_labels = copy.deepcopy(d[key])
            mask_all_labels[mask_all_labels > 0] = 1
            unknown_mask = unknown_mask - mask_all_labels
            unknown_mask[unknown_mask < 0] = 0
            unknown_mask[unknown_mask == 1] = self.max_labels
            d[key] = d[key] + unknown_mask
        return d
