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

from __future__ import annotations

import logging
import os
import threading
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from packaging import version
import numpy as np
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.data.utils import orientation_ras_lps
from monai.utils import MetaKeys, SpaceKeys, TraceKeys, ensure_tuple, optional_import, require_pkg
from torch.utils.data._utils.collate import np_str_obj_array_pattern

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pydicom

    has_pydicom = True
    import cupy as cp

    has_cp = True
    from nvidia import nvimgcodec as nvimgcodec

    has_nvimgcodec = True
else:
    pydicom, has_pydicom = optional_import("pydicom")
    cp, has_cp = optional_import("cupy")
    nvimgcodec, has_nvimgcodec = optional_import("nvidia.nvimgcodec")

logger = logging.getLogger(__name__)

__all__ = ["NvDicomReader"]

# Thread-local storage for nvimgcodec decoder
# Each thread gets its own decoder instance for thread safety
_thread_local = threading.local()


def _get_nvimgcodec_decoder():
    """Get or create a thread-local nvimgcodec decoder singleton."""
    if not has_nvimgcodec:
        raise RuntimeError("nvimgcodec is not available. Cannot create decoder.")
    
    if not hasattr(_thread_local, 'decoder') or _thread_local.decoder is None:
        _thread_local.decoder = nvimgcodec.Decoder()
        logger.debug(f"Initialized thread-local nvimgcodec.Decoder for thread {threading.current_thread().name}")
    
    return _thread_local.decoder


@require_pkg(pkg_name="pydicom")
class NvDicomReader(ImageReader):
    """
    DICOM reader with proper spatial slice ordering.

    This reader properly handles DICOM slice ordering using ImagePositionPatient
    and ImageOrientationPatient tags, ensuring correct 3D volume construction
    for any orientation (axial, sagittal, coronal, or oblique).

    When reading a directory containing multiple series, only the first series
    is read by default (similar to ITKReader behavior).

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata.
        series_name: the SeriesInstanceUID to read when directory contains multiple series.
            If empty (default), reads the first series found.
        series_meta: whether to load series metadata (currently unused).
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS".
            Defaults to ``True``. Set to ``True`` to be consistent with ``NibabelReader``.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False`` (default), returns shape (depth, height, width) following NumPy convention.
            If ``True``, returns shape (width, height, depth) similar to ITK's layout.
            This option does not affect the metadata.
        preserve_dtype: whether to preserve the original DICOM pixel data type after applying rescale.
            If ``True`` (default), converts back to original dtype (matching ITK behavior).
            If ``False``, outputs float32 for all data after rescaling.
        prefer_gpu_output: If True, prefer GPU output over CPU output if the underlying codec supports it. Otherwise, convert to CPU regardless.
            Default is True.
        use_nvimgcodec: If True, use nvImageCodec to decode the pixel data. Default is True. nvImageCodec is required for this option.
            nvImageCodec supports JPEG2000, HTJ2K, and JPEG transfer syntaxes.
        kwargs: additional args for `pydicom.dcmread` API.

    Example:
        >>> # Read first series from directory (default: depth first)
        >>> reader = NvDicomReader()
        >>> img = reader.read("path/to/dicom/dir")
        >>> volume, metadata = reader.get_data(img)
        >>> volume.shape  # (173, 512, 512) = (depth, height, width)
        >>>
        >>> # Read with ITK-style layout (depth last)
        >>> reader = NvDicomReader(reverse_indexing=True)
        >>> img = reader.read("path/to/dicom/dir")
        >>> volume, metadata = reader.get_data(img)
        >>> volume.shape  # (512, 512, 173) = (width, height, depth)
        >>>
        >>> # Output float32 instead of preserving original dtype
        >>> reader = NvDicomReader(preserve_dtype=False)
        >>> img = reader.read("path/to/dicom/dir")
        >>> volume, metadata = reader.get_data(img)
        >>> volume.dtype  # float32 (instead of int32)
        >>>
        >>> # Load to GPU memory with nvImageCodec acceleration
        >>> reader = NvDicomReader(prefer_gpu_output=True, use_nvimgcodec=True)
        >>> img = reader.read("path/to/dicom/dir")
        >>> volume, metadata = reader.get_data(img)
        >>> type(volume).__module__  # 'cupy' (GPU array)
        >>>
        >>> # Read specific series
        >>> reader = NvDicomReader(series_name="1.2.3.4.5.6.7")
        >>> img = reader.read("path/to/dicom/dir")
    """

    def __init__(
        self,
        channel_dim: str | int | None = None,
        series_name: str = "",
        series_meta: bool = False,
        affine_lps_to_ras: bool = True,
        reverse_indexing: bool = False,
        preserve_dtype: bool = True,
        prefer_gpu_output: bool = True,
        use_nvimgcodec: bool = True,
        allow_fallback_decode: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        self.series_name = series_name
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras
        self.reverse_indexing = reverse_indexing
        self.preserve_dtype = preserve_dtype
        self.use_nvimgcodec = use_nvimgcodec
        self.prefer_gpu_output = prefer_gpu_output
        self.allow_fallback_decode = allow_fallback_decode
        # Initialize decode params for nvImageCodec if needed
        if self.use_nvimgcodec:
            if not has_nvimgcodec:
                warnings.warn("NvDicomReader: nvImageCodec not installed, will use pydicom for decoding.")
                self.use_nvimgcodec = False
            else:
                self.decode_params = nvimgcodec.DecodeParams(
                    allow_any_depth=True, color_spec=nvimgcodec.ColorSpec.UNCHANGED
                )

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by NvDicom reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        Returns:
            bool: True if pydicom and nvimgcodec are available and all paths are valid DICOM files or directories containing DICOM files.
        """
        logger.info("verify_suffix: has_pydicom=%s has_nvimgcodec=%s", has_pydicom, has_nvimgcodec)
        if not (has_pydicom and has_nvimgcodec):
            logger.info(
                "verify_suffix: has_pydicom=%s has_nvimgcodec=%s -> returning False", has_pydicom, has_nvimgcodec
            )
            return False

        def _is_dcm_file(path):
            return str(path).lower().endswith(".dcm") and os.path.isfile(str(path))

        def _dir_contains_dcm(path):
            if not os.path.isdir(str(path)):
                return False
            try:
                for f in os.listdir(str(path)):
                    if f.lower().endswith(".dcm") and os.path.isfile(os.path.join(str(path), f)):
                        return True
            except Exception:
                return False
            return False

        paths = ensure_tuple(filename)
        if len(paths) < 1:
            logger.info("verify_suffix: No paths provided.")
            return False

        for fpath in paths:
            if _is_dcm_file(fpath):
                logger.info(f"verify_suffix: Path '{fpath}' is a DICOM file.")
                continue
            elif _dir_contains_dcm(fpath):
                logger.info(f"verify_suffix: Path '{fpath}' is a directory containing at least one DICOM file.")
                continue
            else:
                logger.info(
                    f"verify_suffix: Path '{fpath}' is neither a DICOM file nor a directory containing DICOM files."
                )
                return False
        return True

    def _apply_rescale_and_dtype(self, pixel_data, ds, original_dtype):
        """
        Apply DICOM rescale slope/intercept and handle dtype preservation.
        
        Args:
            pixel_data: numpy or cupy array of pixel data
            ds: pydicom dataset containing RescaleSlope/RescaleIntercept tags
            original_dtype: original dtype before any processing
            
        Returns:
            Processed pixel data array (potentially rescaled and dtype converted)
        """
        # Detect array library (numpy or cupy)
        xp = cp if hasattr(pixel_data, "__cuda_array_interface__") else np
        
        # Check if rescaling is needed
        has_rescale = hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")
        
        if has_rescale:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            slope = xp.asarray(slope, dtype=xp.float32)
            intercept = xp.asarray(intercept, dtype=xp.float32)
            pixel_data = pixel_data.astype(xp.float32) * slope + intercept
            
            # Convert back to original dtype if requested (matching ITK behavior)
            if self.preserve_dtype:
                # Determine target dtype based on original and rescale
                # ITK converts to a dtype that can hold the rescaled values
                # Handle both numpy and cupy dtypes
                orig_dtype_str = str(original_dtype)
                if "uint16" in orig_dtype_str:
                    # uint16 with rescale typically goes to int32 in ITK
                    target_dtype = xp.int32
                elif "int16" in orig_dtype_str:
                    target_dtype = xp.int32
                elif "uint8" in orig_dtype_str:
                    target_dtype = xp.int32
                else:
                    # Preserve original dtype for other types
                    target_dtype = original_dtype
                pixel_data = pixel_data.astype(target_dtype)
        
        return pixel_data

    def _is_nvimgcodec_supported_syntax(self, img):
        """
        Check if the DICOM transfer syntax is supported by nvImageCodec.

        Args:
            img: a Pydicom dataset object.

        Returns:
            bool: True if transfer syntax is supported by nvImageCodec, False otherwise.
        """
        if not has_nvimgcodec:
            return False

        # Check if we have a transfer syntax that nvImageCodec can handle
        file_meta = getattr(img, "file_meta", None)
        if file_meta is None:
            return False

        transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
        if transfer_syntax is None:
            return False

        # Define supported transfer syntaxes for nvImageCodec
        jpeg2000_syntaxes = [
            "1.2.840.10008.1.2.4.90",  # JPEG 2000 Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.91",  # JPEG 2000 Image Compression
        ]

        htj2k_syntaxes = [
            "1.2.840.10008.1.2.4.201",  # High-Throughput JPEG 2000 Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.202",  # High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)
            "1.2.840.10008.1.2.4.203",  # High-Throughput JPEG 2000 Image Compression
        ]

        # JPEG transfer syntaxes (lossy)
        jpeg_lossy_syntaxes = [
            "1.2.840.10008.1.2.4.50",  # JPEG Baseline (Process 1)
            "1.2.840.10008.1.2.4.51",  # JPEG Extended (Process 2 & 4)
        ]

        jpeg_lossless_syntaxes = [
            '1.2.840.10008.1.2.4.57',  # JPEG Lossless, Non-Hierarchical (Process 14)
            '1.2.840.10008.1.2.4.70',  # JPEG Lossless, Non-Hierarchical, First-Order Prediction
        ]

        return str(transfer_syntax) in jpeg2000_syntaxes + htj2k_syntaxes + jpeg_lossy_syntaxes + jpeg_lossless_syntaxes

    def _nvimgcodec_decode(self, img):
        """
        Decode pixel data using nvImageCodec for supported transfer syntaxes.

        Args:
            img: a Pydicom dataset object.

        Returns:
            numpy or cupy array: Decoded pixel data.

        Raises:
            ValueError: If pixel data is missing or decoding fails.
        """
        logger.info(f"NvDicomReader: Starting nvImageCodec decoding")

        # Get raw pixel data
        if not hasattr(img, "PixelData") or img.PixelData is None:
            raise ValueError(f"dicom data: does not have a PixelData member.")

        pixel_data = img.PixelData

        # Decode the pixel data
        data_sequence = [fragment for fragment in pydicom.encaps.generate_frames(pixel_data)]
        logger.info(f"NvDicomReader: Decoding {len(data_sequence)} fragment(s) with nvImageCodec")
        decoder = _get_nvimgcodec_decoder()
        decoder_output = decoder.decode(data_sequence, params=self.decode_params)
        if decoder_output is None:
            raise ValueError(f"nvImageCodec failed to decode")

        # Not all fragments are images, so we need to filter out None images
        decoded_data = [img for img in decoder_output if img is not None]
        if len(decoded_data) == 0:
            raise ValueError(f"nvImageCodec failed to decode or no valid images were found in the decoded data")

        buffer_kind_enum = decoded_data[0].buffer_kind

        # Concatenate all images into a volume if number_of_frames > 1 and multiple images are present
        number_of_frames = getattr(img, "NumberOfFrames", 1)
        if number_of_frames > 1 and len(decoded_data) > 1:
            if number_of_frames != len(decoded_data):
                raise ValueError(
                    f"Number of frames in the image ({number_of_frames}) does not match the number of decoded images ({len(decoded_data)})."
                )
            if buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_DEVICE:
                decoded_array = cp.concatenate([cp.array(d.gpu()) for d in decoded_data], axis=0)
            elif buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_HOST:
                # Use .cpu() to get data from either GPU or CPU buffer
                decoded_array = np.concatenate([np.array(d.cpu()) for d in decoded_data], axis=0)
            else:
                raise ValueError(f"Unknown buffer kind: {buffer_kind_enum}")
        else:
            if buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_DEVICE:
                decoded_array = cp.array(decoded_data[0].cuda())
            elif buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_HOST:
                # Use .cpu() to get data from either GPU or CPU buffer
                decoded_array = np.array(decoded_data[0].cpu())
            else:
                raise ValueError(f"Unknown buffer kind: {buffer_kind_enum}")

        # Reshape based on DICOM parameters
        rows = getattr(img, "Rows", None)
        columns = getattr(img, "Columns", None)
        samples_per_pixel = getattr(img, "SamplesPerPixel", 1)
        number_of_frames = getattr(img, "NumberOfFrames", 1)

        if rows and columns:
            if number_of_frames > 1:
                expected_shape = (number_of_frames, rows, columns)
                if samples_per_pixel > 1:
                    expected_shape = expected_shape + (samples_per_pixel,)
            else:
                expected_shape = (rows, columns)
                if samples_per_pixel > 1:
                    expected_shape = expected_shape + (samples_per_pixel,)

            # Reshape if necessary
            if decoded_array.size == np.prod(expected_shape):
                decoded_array = decoded_array.reshape(expected_shape)

        return decoded_array

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.
        Note that the returned object is ITK image object or list of ITK image objects.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `itk.imread` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

        """
        from pathlib import Path

        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        # Store filenames for later use in get_data (needed for nvImageCodec/GPU loading)
        self.filenames: list = []
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                # read DICOM series
                # Use pydicom to read a DICOM series from the directory `name`.
                logger.info(f"NvDicomReader: Reading DICOM series from directory: {name}")

                # Collect all DICOM files in the directory
                dicom_files = [os.path.join(name, f) for f in os.listdir(name) if os.path.isfile(os.path.join(name, f))]
                if not dicom_files:
                    raise FileNotFoundError(f"No files found in: {name}.")

                # Group files by SeriesInstanceUID and collect metadata
                series_dict = {}
                series_metadata = {}
                logger.info(f"NvDicomReader: Parsing {len(dicom_files)} DICOM files with pydicom")
                for fp in dicom_files:
                    try:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True)
                        if hasattr(ds, "SeriesInstanceUID"):
                            series_uid = ds.SeriesInstanceUID
                            if self.series_name and not series_uid.startswith(self.series_name):
                                continue
                            if series_uid not in series_dict:
                                series_dict[series_uid] = []
                                # Store series metadata from first file
                                series_metadata[series_uid] = {
                                    "SeriesDate": getattr(ds, "SeriesDate", ""),
                                    "SeriesTime": getattr(ds, "SeriesTime", ""),
                                    "SeriesNumber": getattr(ds, "SeriesNumber", 0),
                                    "SeriesDescription": getattr(ds, "SeriesDescription", ""),
                                }
                            series_dict[series_uid].append((fp, ds))
                    except Exception as e:
                        warnings.warn(f"Skipping file {fp}: {e}")

                if self.series_name:
                    if not series_dict:
                        raise FileNotFoundError(
                            f"No valid DICOM series found in {name} matching series name {self.series_name}."
                        )
                elif not series_dict:
                    raise FileNotFoundError(f"No valid DICOM series found in {name}.")

                # Sort series by SeriesDate (and SeriesTime as tiebreaker)
                # This matches ITKReader's behavior with AddSeriesRestriction("0008|0021")
                def series_sort_key(series_uid):
                    meta = series_metadata[series_uid]
                    # Format: (SeriesDate, SeriesTime, SeriesNumber)
                    # Empty strings sort first, so series without dates come first
                    return (meta["SeriesDate"], meta["SeriesTime"], meta["SeriesNumber"])

                sorted_series_uids = sorted(series_dict.keys(), key=series_sort_key)

                # Determine which series to use
                if len(sorted_series_uids) > 1:
                    logger.warning(f"NvDicomReader: Directory {name} contains {len(sorted_series_uids)} DICOM series")

                series_identifier = sorted_series_uids[0] if not self.series_name else self.series_name
                logger.info(f"NvDicomReader: Selected series: {series_identifier}")

                if series_identifier not in series_dict:
                    raise ValueError(
                        f"Series '{series_identifier}' not found in directory. Available series: {sorted_series_uids}"
                    )

                # Get files for the selected series
                series_files = series_dict[series_identifier]

                # Prepare slices with position information for sorting
                slices = []
                slices_without_position = []
                for fp, ds in series_files:
                    if hasattr(ds, "ImagePositionPatient"):
                        pos = np.array(ds.ImagePositionPatient)
                        slices.append((pos, fp, ds))
                    else:
                        # Handle slices without ImagePositionPatient (e.g., localizers, single-slice images)
                        slices_without_position.append((fp, ds))

                if not slices and not slices_without_position:
                    raise FileNotFoundError(f"No readable DICOM slices found in series {series_identifier}.")

                # Sort by spatial position using slice normal projection
                # This works for ANY orientation (axial, sagittal, coronal, oblique)
                if slices:
                    # We have slices with ImagePositionPatient - sort spatially
                    first_ds = slices[0][2]
                    if hasattr(first_ds, "ImageOrientationPatient"):
                        iop = np.array(first_ds.ImageOrientationPatient)
                        row_direction = iop[:3]
                        col_direction = iop[3:]
                        slice_normal = np.cross(row_direction, col_direction)

                        # Project each position onto slice normal and sort by distance
                        slices_with_distance = []
                        for pos, fp, ds in slices:
                            distance = np.dot(pos, slice_normal)
                            slices_with_distance.append((distance, fp, ds))
                        slices_with_distance.sort(key=lambda s: s[0])
                        slices = slices_with_distance
                    else:
                        # Fallback to Z-coordinate if no orientation info
                        slices_with_z = [(pos[2], fp, ds) for pos, fp, ds in slices]
                        slices_with_z.sort(key=lambda s: s[0])
                        slices = slices_with_z

                    # Return sorted list of file paths (not datasets without pixel data)
                    # We'll read the full datasets with pixel data in get_data()
                    sorted_filepaths = [fp for _, fp, _ in slices]
                else:
                    # No ImagePositionPatient - sort by InstanceNumber or keep original order
                    slices_no_pos = []
                    for fp, ds in slices_without_position:
                        inst_num = ds.InstanceNumber if hasattr(ds, "InstanceNumber") else 0
                        slices_no_pos.append((inst_num, fp, ds))
                    slices_no_pos.sort(key=lambda s: s[0])
                    sorted_filepaths = [fp for _, fp, _ in slices_no_pos]
                
                # Read all DICOM files for the series and store as a list of Datasets
                # This allows _process_dicom_series() to handle the series as a whole
                logger.info(f"NvDicomReader: Series contains {len(sorted_filepaths)} slices")
                series_datasets = []
                for fpath in sorted_filepaths:
                    ds = pydicom.dcmread(fpath, **kwargs_)
                    series_datasets.append(ds)
                
                # Append the list of datasets as a single series
                img_.append(series_datasets)
                self.filenames.extend(sorted_filepaths)
            else:
                # Single file
                logger.info(f"NvDicomReader: Parsing single DICOM file with pydicom: {name}")
                ds = pydicom.dcmread(name, **kwargs_)
                img_.append(ds)
                self.filenames.append(name)

        if len(filenames) == 1:
            return img_[0]
        return img_

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded DICOM image(s).

        This function constructs 3D volumes from DICOM series by:
        1. Slices are already sorted by spatial position in read()
        2. Stacking slices into a 3D array
        3. Applying rescale slope/intercept if present
        4. Computing affine matrix for spatial transformations

        Args:
            img: a pydicom dataset object or a list of pydicom dataset objects.

        Returns:
            tuple: (numpy array of image data, metadata dict)
                - Array shape: (depth, height, width) for 3D volumes
                - Metadata contains: affine, spacing, original_affine, spatial_shape
        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}

        # Handle single dataset or list of datasets
        if isinstance(img, pydicom.Dataset):
            datasets = [img]
        elif isinstance(img, list):
            # Check if this is a list of Dataset objects from a DICOM series
            if img and isinstance(img[0], pydicom.Dataset):
                # This is a DICOM series - wrap it so it's processed as one unit
                datasets = [img]
            else:
                # This is a list of something else (shouldn't happen normally)
                datasets = img
        else:
            datasets = ensure_tuple(img)

        for idx, ds_or_list in enumerate(datasets):
            # Check if it's a series (list of datasets) or single dataset
            if isinstance(ds_or_list, list):
                # List of datasets - process as series
                data_array, metadata = self._process_dicom_series(ds_or_list)
            elif isinstance(ds_or_list, pydicom.Dataset):
                data_array = self._get_array_data(ds_or_list)
                metadata = self._get_meta_dict(ds_or_list)
                metadata[MetaKeys.SPATIAL_SHAPE] = np.asarray(data_array.shape)

            img_array.append(data_array)
            metadata[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(metadata, self.affine_lps_to_ras)
            metadata[MetaKeys.AFFINE] = metadata[MetaKeys.ORIGINAL_AFFINE].copy()
            metadata[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS

            if self.channel_dim is None:
                metadata[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data_array.shape) == len(metadata[MetaKeys.SPATIAL_SHAPE]) else -1
                )
            else:
                metadata[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim

            _copy_compatible_dict(metadata, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _process_dicom_series(self, datasets: list) -> tuple[np.ndarray, dict]:
        """
        Process a list of sorted DICOM Dataset objects into a 3D volume.

        This method implements batch decoding optimization: when all files use
        nvImageCodec-supported transfer syntaxes, all frames are decoded in a
        single nvImageCodec call for better performance. Falls back to
        frame-by-frame decoding if batch decode fails or is not applicable.

        Args:
            datasets: list of pydicom Dataset objects (already sorted by spatial position)

        Returns:
            tuple: (3D numpy array, metadata dict)
        """
        if not datasets:
            raise ValueError("Empty dataset list")

        first_ds = datasets[0]
        needs_rescale = hasattr(first_ds, "RescaleSlope") and hasattr(first_ds, "RescaleIntercept")
        rows = first_ds.Rows
        cols = first_ds.Columns
        depth = len(datasets)

        # Check if we can use nvImageCodec on the whole series
        can_use_nvimgcodec = self.use_nvimgcodec and all(self._is_nvimgcodec_supported_syntax(ds) for ds in datasets)

        batch_decode_success = False
        original_dtype = None

        if can_use_nvimgcodec:
            logger.info(f"NvDicomReader: Using nvImageCodec batch decode for {depth} slices")
            try:
                # Batch decode all frames in a single nvImageCodec call
                # Collect all compressed frames from all DICOM files
                all_frames = []
                for ds in datasets:
                    if not hasattr(ds, "PixelData") or ds.PixelData is None:
                        raise ValueError("DICOM data does not have pixel data")
                    pixel_data = ds.PixelData
                    # Extract compressed frame(s) from this DICOM file
                    frames = [fragment for fragment in pydicom.encaps.generate_frames(pixel_data)]
                    all_frames.extend(frames)

                # Decode all frames at once
                decoder = _get_nvimgcodec_decoder()
                decoded_data = decoder.decode(all_frames, params=self.decode_params)

                if not decoded_data or any(d is None for d in decoded_data):
                    raise ValueError("nvImageCodec batch decode failed")

                # Determine buffer location (GPU or CPU)
                buffer_kind_enum = decoded_data[0].buffer_kind

                # Convert all decoded frames to numpy/cupy arrays
                if buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_DEVICE:
                    xp = cp
                    decoded_arrays = [cp.array(d.cuda()) for d in decoded_data]
                elif buffer_kind_enum == nvimgcodec.ImageBufferKind.STRIDED_HOST:
                    xp = np
                    decoded_arrays = [np.array(d.cpu()) for d in decoded_data]
                else:
                    raise ValueError(f"Unknown buffer kind: {buffer_kind_enum}")

                original_dtype = decoded_arrays[0].dtype
                dtype_vol = xp.float32 if needs_rescale else original_dtype

                # Build 3D volume (use float32 for rescaling to avoid overflow)
                # Shape depends on reverse_indexing
                if self.reverse_indexing:
                    volume = xp.zeros((cols, rows, depth), dtype=dtype_vol)
                else:
                    volume = xp.zeros((depth, rows, cols), dtype=dtype_vol)

                for frame_idx, frame_array in enumerate(decoded_arrays):
                    # Reshape if needed
                    if frame_array.shape != (rows, cols):
                        frame_array = frame_array.reshape(rows, cols)

                    if self.reverse_indexing:
                        volume[:, :, frame_idx] = frame_array.T
                    else:
                        volume[frame_idx, :, :] = frame_array

                batch_decode_success = True

            except Exception as e:
                if not self.allow_fallback_decode:
                    raise ValueError(f"nvImageCodec batch decoding failed: {e}")
                warnings.warn(f"nvImageCodec batch decoding failed: {e}. Falling back to frame-by-frame.")
                batch_decode_success = False

        if not batch_decode_success or not can_use_nvimgcodec:
            # Fallback: use pydicom pixel_array for each frame
            logger.info(f"NvDicomReader: Using pydicom pixel_array decode for {depth} slices")
            first_pixel_array = first_ds.pixel_array
            original_dtype = first_pixel_array.dtype

            # Build 3D volume (use float32 for rescaling to avoid overflow if needed)
            xp = cp if hasattr(first_pixel_array, "__cuda_array_interface__") else np
            dtype_vol = xp.float32 if needs_rescale else original_dtype

            # Shape depends on reverse_indexing
            if self.reverse_indexing:
                volume = xp.zeros((cols, rows, depth), dtype=dtype_vol)
            else:
                volume = xp.zeros((depth, rows, cols), dtype=dtype_vol)

            for frame_idx, ds in enumerate(datasets):
                frame_array = ds.pixel_array
                # Ensure correct array type
                if hasattr(frame_array, "__cuda_array_interface__"):
                    frame_array = cp.asarray(frame_array)
                else:
                    frame_array = np.asarray(frame_array)

                if self.reverse_indexing:
                    volume[:, :, frame_idx] = frame_array.T
                else:
                    volume[frame_idx, :, :] = frame_array

        # Ensure xp is defined for subsequent operations
        xp = cp if hasattr(volume, "__cuda_array_interface__") else np

        # Ensure original_dtype is set
        if original_dtype is None:
            # Get dtype from first pixel array if not already set
            original_dtype = first_ds.pixel_array.dtype

        # Apply rescaling and dtype conversion using common helper
        volume = self._apply_rescale_and_dtype(volume, first_ds, original_dtype)

        # Calculate spacing
        pixel_spacing = first_ds.PixelSpacing if hasattr(first_ds, "PixelSpacing") else [1.0, 1.0]

        # Calculate slice spacing
        if depth > 1:
            # Prioritize calculating from actual slice positions (more accurate than SliceThickness tag)
            # This matches ITKReader behavior and handles cases where SliceThickness != actual spacing
            if hasattr(first_ds, "ImagePositionPatient"):
                # Calculate average distance between consecutive slices using z-coordinate
                # This matches ITKReader's approach (see lines 595-612)
                average_distance = 0.0
                prev_pos = np.array(datasets[0].ImagePositionPatient)[2]
                for i in range(1, len(datasets)):
                    if hasattr(datasets[i], "ImagePositionPatient"):
                        curr_pos = np.array(datasets[i].ImagePositionPatient)[2]
                        average_distance += abs(curr_pos - prev_pos)
                        prev_pos = curr_pos
                slice_spacing = average_distance / (len(datasets) - 1)
            elif hasattr(first_ds, "SliceThickness"):
                # Fallback to SliceThickness tag if positions unavailable
                slice_spacing = float(first_ds.SliceThickness)
            else:
                slice_spacing = 1.0
        else:
            slice_spacing = 1.0

        # Build metadata
        metadata = self._get_meta_dict(first_ds)
        metadata["spacing"] = np.array([float(pixel_spacing[1]), float(pixel_spacing[0]), slice_spacing])
        # Metadata should always use numpy arrays, even if data is on GPU
        metadata[MetaKeys.SPATIAL_SHAPE] = np.asarray(volume.shape)

        # Store last position for affine calculation
        if hasattr(datasets[-1], "ImagePositionPatient"):
            metadata["lastImagePositionPatient"] = np.array(datasets[-1].ImagePositionPatient)

        return volume, metadata

    def _get_array_data(self, ds):
        """
        Get pixel array from a single DICOM dataset.

        Args:
            ds: pydicom dataset object

        Returns:
            numpy or cupy array of pixel data
        """
        # Get pixel array using nvImageCodec or GPU loading if enabled and filename available
        if self.use_nvimgcodec and self._is_nvimgcodec_supported_syntax(ds):
            try:
                pixel_array = self._nvimgcodec_decode(ds)
                original_dtype = pixel_array.dtype
                logger.info(f"NvDicomReader: Successfully decoded with nvImageCodec")
            except Exception as e:
                logger.warning(
                    f"NvDicomReader: nvImageCodec decoding failed: {e}, falling back to pydicom"
                )
                pixel_array = ds.pixel_array
                original_dtype = pixel_array.dtype
        else:
            logger.info(f"NvDicomReader: Using pydicom pixel_array decode")
            pixel_array = ds.pixel_array
            original_dtype = pixel_array.dtype

        # Apply rescaling and dtype conversion using common helper
        pixel_array = self._apply_rescale_and_dtype(pixel_array, ds, original_dtype)

        return pixel_array

    def _get_meta_dict(self, ds) -> dict:
        """Extract metadata from DICOM dataset, storing all tags like ITKReader does."""
        metadata = {}

        # Store all DICOM tags in ITK format (GGGG|EEEE)
        for elem in ds:
            # Skip pixel data and large binary data
            if elem.tag in [
                (0x7FE0, 0x0010),  # Pixel Data
                (0x7FE0, 0x0008),  # Float Pixel Data
                (0x7FE0, 0x0009),
            ]:  # Double Float Pixel Data
                continue

            # Format tag as 'GGGG|EEEE' (matching ITK format)
            tag_str = f"{elem.tag.group:04x}|{elem.tag.element:04x}"

            # Store the value, converting to appropriate Python types
            if elem.VR == "SQ":  # Sequence - skip for now (can be very large)
                continue
            try:
                # Convert value to appropriate Python type
                value = elem.value

                # Handle pydicom special types
                value_type_name = type(value).__name__
                if value_type_name == "MultiValue":
                    # MultiValue: convert to list
                    value = list(value)
                elif value_type_name == "PersonName":
                    # PersonName: convert to string
                    value = str(value)
                elif hasattr(value, "tolist"):
                    # NumPy arrays: convert to list or scalar
                    value = value.tolist() if value.size > 1 else value.item()
                elif isinstance(value, bytes):
                    # Bytes: decode to string
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except:
                        value = str(value)

                metadata[tag_str] = value
            except Exception:
                # Some values might not be decodable, skip them
                pass

        # Also store essential spatial tags with readable names
        # (for convenience and backward compatibility)
        if hasattr(ds, "ImageOrientationPatient"):
            metadata["ImageOrientationPatient"] = list(ds.ImageOrientationPatient)
        if hasattr(ds, "ImagePositionPatient"):
            metadata["ImagePositionPatient"] = list(ds.ImagePositionPatient)
        if hasattr(ds, "PixelSpacing"):
            metadata["PixelSpacing"] = list(ds.PixelSpacing)

        return metadata

    def _get_affine(self, metadata: dict, lps_to_ras: bool = True) -> np.ndarray:
        """
        Construct affine matrix from DICOM metadata.

        Args:
            metadata: metadata dictionary
            lps_to_ras: whether to convert from LPS to RAS

        Returns:
            4x4 affine matrix
        """
        affine = np.eye(4)

        if "ImageOrientationPatient" not in metadata or "ImagePositionPatient" not in metadata:
            # No explicit orientation info - use identity but still apply LPS->RAS if requested
            # DICOM default coordinate system is LPS
            if lps_to_ras:
                affine = orientation_ras_lps(affine)
            return affine

        iop = metadata["ImageOrientationPatient"]
        ipp = metadata["ImagePositionPatient"]
        spacing = metadata.get("spacing", np.array([1.0, 1.0, 1.0]))

        # Extract direction cosines
        row_cosine = np.array(iop[:3])
        col_cosine = np.array(iop[3:])

        # Build affine matrix
        # Column 0: row direction * row spacing
        affine[:3, 0] = row_cosine * spacing[0]
        # Column 1: col direction * col spacing
        affine[:3, 1] = col_cosine * spacing[1]

        # Calculate slice direction
        # Determine the depth dimension (handle reverse_indexing)
        spatial_shape = metadata[MetaKeys.SPATIAL_SHAPE]
        if len(spatial_shape) == 3:
            # Find which dimension is the depth (smallest for typical medical images)
            # When reverse_indexing=True: shape is (W, H, D), depth is at index 2
            # When reverse_indexing=False: shape is (D, H, W), depth is at index 0
            depth_idx = np.argmin(spatial_shape)
            n_slices = spatial_shape[depth_idx]

            if n_slices > 1 and "lastImagePositionPatient" in metadata:
                # Multi-slice: calculate from first and last positions
                last_ipp = metadata["lastImagePositionPatient"]
                slice_vec = (last_ipp - np.array(ipp)) / (n_slices - 1)
                affine[:3, 2] = slice_vec
            else:
                # Single slice or no last position: use cross product
                slice_normal = np.cross(row_cosine, col_cosine)
                affine[:3, 2] = slice_normal * spacing[2]
        else:
            # 2D image - use cross product
            slice_normal = np.cross(row_cosine, col_cosine)
            affine[:3, 2] = slice_normal * spacing[2]

        # Translation
        affine[:3, 3] = ipp

        # Convert LPS to RAS if requested
        if lps_to_ras:
            affine = orientation_ras_lps(affine)

        return affine
