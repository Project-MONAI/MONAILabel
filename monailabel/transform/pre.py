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
from typing import Optional

from monai.config import KeysCollection
from monai.data import ImageReader, MetaTensor
from monai.transforms import LoadImaged, MapTransform

from monai.utils.enums import ColorOrder

import inspect
import logging
import sys
import traceback
import warnings
from pathlib import Path
from pydoc import locate
from typing import Dict, List, Optional, Sequence, Type, Union, TYPE_CHECKING

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data import image_writer
from monai.data.folder_layout import FolderLayout

from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import Transform
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import GridSamplePadMode, ensure_tuple_rep
from monai.utils import ImageMetaKey as Key
from monai.utils import OptionalImportError, convert_to_dst_type, ensure_tuple, look_up_option, optional_import
from monai.utils.enums import PostFix
import os

if TYPE_CHECKING:
    import cv2

    has_cv2 = True
else:
    cv2, has_cv2 = optional_import("cv2")

logger = logging.getLogger(__name__)

DEFAULT_POST_FIX = PostFix.meta()


class LoadImageExd(LoadImaged):
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)

        ignore = False
        for i, key in enumerate(self.keys):
            # Support direct image in np (pass only transform)
            if not isinstance(d[key], str):
                ignore = True
                meta_dict_key = f"{key}_{self.meta_key_postfix[i]}"
                meta_dict = d.get(meta_dict_key)
                if meta_dict is None:
                    d[meta_dict_key] = dict()
                    meta_dict = d.get(meta_dict_key)

                image_np = d[key]
                meta_dict["spatial_shape"] = image_np.shape[:-1]  # type: ignore
                meta_dict["original_channel_dim"] = -1  # type: ignore
                meta_dict["original_affine"] = None  # type: ignore

                d[key] = MetaTensor(image_np, meta=meta_dict)
                continue

        if not ignore:
            d = super().__call__(d, reader)

        return d


class NormalizeLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, value=1) -> None:
        super().__init__(keys, allow_missing_keys)
        self.value = value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key].array
            label[label > 0] = self.value
            d[key].array = label
        return d




class SuppressStderr:
    """Suppress stderr. Useful as OpenCV (and dependencies) can produce a lot of output."""

    def __enter__(self):
        self.errnull_file = open(os.devnull, "w")
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stderr = sys.stderr
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stderr_fileno)
        self.errnull_file.close()

class LoadVideoFrame(Transform):
    def __init__(
        self,
        max_num_frames: Optional[int] = None,
        dtype: DtypeLike = np.float32,
        color_order: str = ColorOrder.RGB,
        channel_dim: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """
        Base video dataset.
        """
        if not has_cv2:
            raise RuntimeError("OpenCV not installed.")
        if color_order not in ColorOrder:
            raise NotImplementedError

        self.color_order = color_order
        self.channel_dim = channel_dim
        self.dtype = dtype
        self.max_num_frames = max_num_frames

    def __call__(self, video_source: Union[str, int], frame_id: int = 0):

        """
        Use CV2 for loading videos and retrive frame image by frame index
        """

        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        with SuppressStderr():
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_source}")

        cap.set(1,frame_id); # Where frame_no is the frame you want

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to read frame.")
        # Switch color order if desired
        if self.color_order == ColorOrder.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = convert_to_dst_type(frame, dst=frame, dtype=self.dtype)[0]

        return frame


class LoadVideoFramed(MapTransform):
    """

    """

    def __init__(
        self,
        keys: KeysCollection,
        max_num_frames: Optional[int] = None,
        dtype: DtypeLike = np.float32,
        color_order: str = ColorOrder.RGB,
        channel_dim: int = 0,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadVideoFrame(
            max_num_frames,
            dtype,
            color_order,
            channel_dim,
            *args,
            **kwargs,
        )
        # Meta tensor is not implemented for videos yet, remain these meta keys as TODO
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data, frame_id: int = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        frame_id = d['frame_id'] if frame_id == None else frame_id
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], frame_id)
            d[key] = data

        return d












