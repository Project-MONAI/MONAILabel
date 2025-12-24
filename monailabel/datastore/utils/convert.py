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

import datetime
import json
import logging
import os
import pathlib
import tempfile
import time

import numpy as np
import pydicom
import pydicom.errors
import SimpleITK
from monai.transforms import LoadImage
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

try:
    import highdicom as hd

    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False

from monailabel import __version__
from monailabel.config import settings
from monailabel.datastore.utils.colors import GENERIC_ANATOMY_COLORS
from monailabel.transform.writer import write_itk

logger = logging.getLogger(__name__)


def dicom_to_nifti(series_dir, is_seg=False):
    start = time.time()

    if is_seg:
        output_file = dicom_seg_to_itk_image(series_dir)
    else:
        # https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
        if os.path.isdir(series_dir) and len(os.listdir(series_dir)) > 1:
            reader = SimpleITK.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            filename = (
                series_dir if not os.path.isdir(series_dir) else os.path.join(series_dir, os.listdir(series_dir)[0])
            )

            file_reader = SimpleITK.ImageFileReader()
            file_reader.SetImageIO("GDCMImageIO")
            file_reader.SetFileName(filename)
            image = file_reader.Execute()

        logger.info(f"Image size: {image.GetSize()}")
        output_file = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        SimpleITK.WriteImage(image, output_file)

    logger.info(f"dicom_to_nifti latency : {time.time() - start} (sec)")
    return output_file


def binary_to_image(reference_image, label, dtype=np.uint8, file_ext=".nii.gz"):
    start = time.time()

    image_np, meta_dict = LoadImage(image_only=False)(reference_image)
    label_np = np.fromfile(label, dtype=dtype)

    logger.info(f"Image: {image_np.shape}")
    logger.info(f"Label: {label_np.shape}")

    label_np = label_np.reshape(image_np.shape, order="F")
    logger.info(f"Label (reshape): {label_np.shape}")

    output_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
    affine = meta_dict.get("affine")
    write_itk(label_np, output_file, affine=affine, dtype=None, compress=True)

    logger.info(f"binary_to_image latency : {time.time() - start} (sec)")
    return output_file


def _extract_label_info(label, label_info):
    """Extract unique labels and model info from label file.

    Args:
        label: Path to NIfTI label file
        label_info: List of dictionaries containing segment information, each with optional
                   "idx"/"labelID"/"label" field to map to actual label values

    Returns:
        tuple: (unique_labels, info_by_label_id, model_name) or (None, None, None) if empty
    """
    # Load label file using SimpleITK (consistent with conversion pipeline)
    mask = SimpleITK.ReadImage(label)
    label_array = SimpleITK.GetArrayFromImage(mask)

    # Extract unique non-zero labels
    unique_labels = np.unique(label_array).astype(np.int_)
    unique_labels = unique_labels[unique_labels != 0]

    if not unique_labels.size:
        logger.warning("No non-zero labels found in segmentation")
        return None, None, None

    # Build mapping from label ID to metadata
    # Look for explicit ID fields: "idx", "labelID", or "label"
    info_by_label_id = {}
    has_explicit_ids = False

    if label_info:
        for entry in label_info:
            # Find the label ID from various possible field names
            label_id = entry.get("idx") or entry.get("labelID") or entry.get("label")
            if label_id is not None:
                info_by_label_id[int(label_id)] = entry
                has_explicit_ids = True

        # If no explicit IDs found, fall back to positional mapping
        # Assume label_info is ordered to match unique_labels
        if not has_explicit_ids:
            for i, label_id in enumerate(unique_labels):
                if i < len(label_info):
                    info_by_label_id[int(label_id)] = label_info[i]

    # Extract model_name (can be in any entry, prefer first entry)
    model_name = "MONAILabel"  # Default
    if label_info and len(label_info) > 0:
        model_name = label_info[0].get("model_name", "MONAILabel")

    return unique_labels, info_by_label_id, model_name


def _highdicom_nifti_to_dicom_seg(
    series_dir, label, label_info, file_ext="*", omit_empty_frames=False, custom_tags=None
) -> str:
    """Convert NIfTI segmentation to DICOM SEG format using highdicom.

    Args:
        series_dir: Directory containing source DICOM images
        label: Path to NIfTI label file
        label_info: List of dictionaries containing segment information (name, description, color, etc.)
        file_ext: File extension pattern for DICOM files (default: "*")
        omit_empty_frames: If True, omit frames with no segmented pixels (default: False)
        custom_tags: Optional dictionary of custom DICOM tags to add (keyword: value)

    Returns:
        Path to output DICOM SEG file, or empty string if conversion fails
    """
    # Input validation
    if label_info is None:
        label_info = []

    if not os.path.exists(label):
        logger.error(f"Label file not found: {label}")
        return ""

    if not os.path.exists(series_dir):
        logger.error(f"Series directory not found: {series_dir}")
        return ""

    # Extract label information
    unique_labels, info_by_label_id, model_name = _extract_label_info(label, label_info)
    if unique_labels is None:
        return ""

    # Build highdicom segment descriptions
    segment_descriptions = []
    for i, label_id in enumerate(unique_labels):
        # Look up metadata by actual label ID, fall back to empty dict
        info = info_by_label_id.get(int(label_id), {})
        name = info.get("name", f"Segment_{label_id}")
        logger.info(f"Segment {i}: idx={label_id}, name={name}")

        # Get category code from label_info or use default
        category_code_dict = info.get("SegmentedPropertyCategoryCodeSequence", {})
        if category_code_dict and isinstance(category_code_dict, dict):
            category_code = Code(
                value=category_code_dict.get("CodeValue", "123037004"),
                scheme_designator=category_code_dict.get("CodingSchemeDesignator", "SCT"),
                meaning=category_code_dict.get("CodeMeaning", "Anatomical Structure"),
            )
        else:
            category_code = codes.SCT.Organ  # Default: Organ

        # Get type code from label_info or use default
        type_code_dict = info.get("SegmentedPropertyTypeCodeSequence", {})
        if type_code_dict and isinstance(type_code_dict, dict):
            type_code = Code(
                value=type_code_dict.get("CodeValue", "78961009"),
                scheme_designator=type_code_dict.get("CodingSchemeDesignator", "SCT"),
                meaning=type_code_dict.get("CodeMeaning", name),
            )
        else:
            # Default type code
            type_code = Code("78961009", "SCT", name)

        # Create highdicom segment description
        # Use sequential segment numbers (1, 2, 3...) to match remapped pixel array
        seg_desc = hd.seg.SegmentDescription(
            segment_number=i + 1,  # Sequential numbering: 1, 2, 3...
            segment_label=name,
            segmented_property_category=category_code,
            segmented_property_type=type_code,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="MONAILABEL", family=codes.DCM.ArtificialIntelligence, version=model_name
            ),
            algorithm_type="AUTOMATIC",
        )
        segment_descriptions.append(seg_desc)

    if not segment_descriptions:
        logger.error("Missing segment descriptions")
        return ""

    # Read source DICOM images (headers only for memory efficiency)
    series_dir = pathlib.Path(series_dir)
    image_files = list(series_dir.glob(file_ext))

    # Read DICOM files with error handling for non-DICOM files
    image_datasets = []
    for f in image_files:
        try:
            ds = dcmread(str(f), stop_before_pixels=True)
            image_datasets.append(ds)
        except (pydicom.errors.InvalidDicomError, OSError, ValueError) as e:
            logger.warning(f"Skipping non-DICOM or invalid file {f}: {e}")
            continue

    if not image_datasets:
        logger.warning(f"No DICOM images found in {series_dir} with pattern {file_ext}")
        return ""

    # Spatially sort DICOM images for correct slice ordering
    # Use ImageOrientationPatient and ImagePositionPatient for robust sorting
    def spatial_sort_key(ds):
        """Generate sort key based on spatial position."""
        try:
            # Get image orientation (row and column direction cosines)
            iop = ds.ImageOrientationPatient
            row_dir = np.array(iop[0:3])
            col_dir = np.array(iop[3:6])

            # Compute plane normal (perpendicular to image plane)
            normal = np.cross(row_dir, col_dir)

            # Get image position (origin of slice)
            position = np.array(ds.ImagePositionPatient)

            # Project position onto normal direction (gives slice location)
            slice_location = np.dot(position, normal)
            return slice_location
        except (AttributeError, KeyError, TypeError):
            # Fall back to InstanceNumber if spatial attributes missing
            try:
                return float(ds.InstanceNumber)
            except (AttributeError, KeyError, TypeError):
                # Final fallback: use 0 (will maintain original order)
                return 0.0

    image_datasets = sorted(image_datasets, key=spatial_sort_key)
    logger.info(f"Total Source Images: {len(image_datasets)}")

    # Load label using SimpleITK for correct axis ordering (D, H, W)
    # SimpleITK natively gives (D, H, W) which matches DICOM/highdicom expectations
    mask = SimpleITK.ReadImage(label)
    mask = SimpleITK.Cast(mask, SimpleITK.sitkUInt16)  # Support up to 65,535 segments
    seg_array = SimpleITK.GetArrayFromImage(mask)

    # Remap label values to sequential 1, 2, 3... as required by DICOM SEG
    # Value 0 is reserved for background (no segment)
    remapped_array = np.zeros_like(seg_array, dtype=np.uint16)
    for new_idx, orig_idx in enumerate(unique_labels, start=1):
        remapped_array[seg_array == orig_idx] = new_idx
    seg_array = remapped_array

    # Get software version
    try:
        software_version = f"MONAI Label {__version__}"
    except (AttributeError, NameError):
        software_version = "MONAI Label"

    # Get consistent timestamp for all DICOM attributes
    dt_now = datetime.datetime.now()

    # DICOM series number is 4 digits max (0-9999)
    MAX_SERIES_NUMBER = 9999
    series_number = int(dt_now.strftime("%H%M%S")) % (MAX_SERIES_NUMBER + 1)

    # Create DICOM SEG using highdicom
    # Use LABELMAP type for indexed labelmap (integer array with values 0..N)
    # BINARY type requires 4D one-hot encoding (F, H, W, S)
    seg = hd.seg.Segmentation(
        source_images=image_datasets,
        pixel_array=seg_array,
        segmentation_type=hd.seg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="MONAI Consortium",
        manufacturer_model_name="MONAI Label",
        software_versions=software_version,
        device_serial_number="0000",
        omit_empty_frames=omit_empty_frames,
    )

    # Add timestamp and timezone
    seg.SeriesDate = dt_now.strftime("%Y%m%d")
    seg.SeriesTime = dt_now.strftime("%H%M%S")

    # Compute timezone offset in ±HHMM format
    def format_timezone_offset(dt):
        """Compute timezone offset from UTC in ±HHMM format."""
        offset = dt.utcoffset()
        if offset is None:
            return "+0000"

        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        abs_seconds = abs(total_seconds)
        hours = abs_seconds // 3600
        minutes = (abs_seconds % 3600) // 60
        return f"{sign}{hours:02d}{minutes:02d}"

    seg.TimezoneOffsetFromUTC = format_timezone_offset(dt_now)
    seg.SeriesDescription = model_name

    # Add Contributing Equipment Sequence
    try:
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence as PyDicomSequence

        # Create Purpose of Reference Code Sequence
        seq_purpose_of_reference_code = PyDicomSequence()
        seg_purpose_of_reference_code = Dataset()
        seg_purpose_of_reference_code.CodeValue = "Newcode1"
        seg_purpose_of_reference_code.CodingSchemeDesignator = "99IHE"
        seg_purpose_of_reference_code.CodeMeaning = "Processing Algorithm"
        seq_purpose_of_reference_code.append(seg_purpose_of_reference_code)

        # Create Contributing Equipment Sequence
        seq_contributing_equipment = PyDicomSequence()
        seg_contributing_equipment = Dataset()
        seg_contributing_equipment.PurposeOfReferenceCodeSequence = seq_purpose_of_reference_code
        seg_contributing_equipment.Manufacturer = "MONAI Consortium"
        seg_contributing_equipment.ManufacturerModelName = model_name
        seg_contributing_equipment.SoftwareVersions = software_version
        seg_contributing_equipment.DeviceUID = hd.UID()
        seq_contributing_equipment.append(seg_contributing_equipment)
        seg.ContributingEquipmentSequence = seq_contributing_equipment
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning(f"Could not add ContributingEquipmentSequence: {e}")

    # Add custom tags if provided
    if custom_tags:
        from pydicom.datadict import tag_for_keyword

        for keyword, value in custom_tags.items():
            if not isinstance(keyword, str):
                logger.warning(f"Custom tag key must be a DICOM keyword string; got {type(keyword)}")
                continue
            try:
                if tag_for_keyword(keyword) is None:
                    logger.warning(f"Unknown DICOM keyword: {keyword}; skipping")
                    continue
                setattr(seg, keyword, value)
            except (AttributeError, KeyError, TypeError, pydicom.errors.InvalidDicomError) as ex:
                logger.exception(f"Custom tag {keyword} was not written")
                continue

    # Save DICOM SEG
    output_file = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False).name
    seg.save_as(output_file)
    logger.info(f"DICOM SEG saved to: {output_file}")

    return output_file


def _itk_nifti_to_dicom_seg(series_dir, label, label_info) -> str:
    """Convert NIfTI segmentation to DICOM SEG format using ITK/dcmqi.

    Args:
        series_dir: Directory containing source DICOM images
        label: Path to NIfTI label file
        label_info: List of dictionaries containing segment information (name, description, color, etc.)

    Returns:
        Path to output DICOM SEG file, or empty string if conversion fails
    """
    # Input validation
    if label_info is None:
        label_info = []

    if not os.path.exists(label):
        logger.error(f"Label file not found: {label}")
        return ""

    if not os.path.exists(series_dir):
        logger.error(f"Series directory not found: {series_dir}")
        return ""

    # Extract label information (reuse helper function)
    unique_labels, info_by_label_id, model_name = _extract_label_info(label, label_info)
    if unique_labels is None:
        return ""

    # Build ITK segment descriptions
    segment_descriptions = []
    for i, label_id in enumerate(unique_labels):
        # Look up metadata by actual label ID, fall back to empty dict
        info = info_by_label_id.get(int(label_id), {})
        name = info.get("name", f"Segment_{label_id}")
        description = info.get("description", name)

        logger.info(f"Segment {i}: idx={label_id}, name={name}")

        # Check if custom segmentAttribute is provided
        segment_attr = info.get("segmentAttribute")

        if segment_attr:
            # Use custom attribute as-is
            segment_descriptions.append(segment_attr)
        else:
            # Build default template for ITK method
            rgb = list(info.get("color", GENERIC_ANATOMY_COLORS.get(name, (255, 0, 0))))[0:3]
            rgb = [int(x) for x in rgb]

            segment_attr = {
                "labelID": int(label_id),
                "SegmentLabel": name,
                "SegmentDescription": description,
                "SegmentAlgorithmType": "AUTOMATIC",
                "SegmentAlgorithmName": "MONAILABEL",
                "SegmentedPropertyCategoryCodeSequence": {
                    "CodeValue": "123037004",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Anatomical Structure",
                },
                "SegmentedPropertyTypeCodeSequence": {
                    "CodeValue": "78961009",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": name,
                },
                "recommendedDisplayRGBValue": rgb,
            }
            segment_descriptions.append(segment_attr)

    if not segment_descriptions:
        logger.error("Missing segment descriptions")
        return ""

    # Extract metadata from label_info (use first segment's metadata for study-level info)
    first_info = label_info[0] if label_info and len(label_info) > 0 else {}

    # Get timestamp-based series number (consistent with highdicom implementation)
    dt_now = datetime.datetime.now()
    MAX_SERIES_NUMBER = 9999
    series_number = int(dt_now.strftime("%H%M%S")) % (MAX_SERIES_NUMBER + 1)

    # Build ITK template with extracted or sensible default values
    template = {
        "ContentCreatorName": first_info.get("creator", "MONAI Label"),
        "ClinicalTrialSeriesID": first_info.get("session_id", "Session1"),
        "ClinicalTrialTimePointID": first_info.get("timepoint_id", "1"),
        "SeriesDescription": model_name,
        "SeriesNumber": str(series_number),  # Use timestamp-based number
        "InstanceNumber": "1",
        "segmentAttributes": [segment_descriptions],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "MONAI Label - Image segmentation",
        "ClinicalTrialCoordinatingCenterName": "MONAI",
        "BodyPartExamined": first_info.get("body_part", ""),
    }
    logger.debug("dcmqi template: %s", json.dumps(template, indent=2))

    # Call dcmqi converter with error handling
    try:
        return _dcmqi_nifti_to_dicom_seg(label, series_dir, template)
    except (RuntimeError, OSError, ValueError) as e:
        logger.exception("ITK DICOM SEG conversion failed")
        return ""


def nifti_to_dicom_seg(
    series_dir, label, label_info, file_ext="*", use_itk=None, omit_empty_frames=False, custom_tags=None
) -> str:
    """Convert NIfTI segmentation to DICOM SEG format.

    This dispatcher function selects between highdicom (default) or ITK/dcmqi implementations.

    Args:
        series_dir: Directory containing source DICOM images
        label: Path to NIfTI label file
        label_info: List of dictionaries containing segment information
        file_ext: File extension pattern for DICOM files (default: "*")
        use_itk: If True, use ITK/dcmqi. If False/None, use highdicom (default).
        omit_empty_frames: If True, omit frames with no segmented pixels (highdicom only)
        custom_tags: Optional dictionary of custom DICOM tags (highdicom only)

    Returns:
        Path to output DICOM SEG file, or empty string if conversion fails
    """
    start = time.time()

    # Determine which implementation to use
    if use_itk is None:
        use_itk = settings.MONAI_LABEL_USE_ITK_FOR_DICOM_SEG

    # Check if highdicom is available (unless using ITK)
    if not use_itk and not HIGHDICOM_AVAILABLE:
        raise ImportError("highdicom is not available")

    # Dispatch to appropriate implementation
    if use_itk:
        output_file = _itk_nifti_to_dicom_seg(series_dir, label, label_info)
    else:
        output_file = _highdicom_nifti_to_dicom_seg(
            series_dir, label, label_info, file_ext, omit_empty_frames, custom_tags
        )

    logger.info(f"nifti_to_dicom_seg latency: {time.time() - start:.3f} sec")
    return output_file


def _dcmqi_nifti_to_dicom_seg(label, series_dir, template) -> str:
    """Convert NIfTI to DICOM SEG using dcmqi's itkimage2segimage command-line tool.

    This is a low-level wrapper around the dcmqi itkimage2segimage tool.
    Called by _itk_nifti_to_dicom_seg() as the actual conversion implementation.
    """
    import shutil

    from monailabel.utils.others.generic import run_command

    command = "itkimage2segimage"
    if not shutil.which(command):
        error_msg = (
            f"\n{'=' * 80}\n"
            f"ERROR: {command} command-line tool not found\n"
            f"{'=' * 80}\n\n"
            f"The ITK-based DICOM SEG conversion requires the dcmqi package.\n\n"
            f"Install dcmqi:\n"
            f"  pip install dcmqi\n\n"
            f"For more information:\n"
            f"  https://github.com/QIICR/dcmqi\n\n"
            f"Note: Consider using the default highdicom-based conversion (use_itk=False)\n"
            f"which doesn't require dcmqi.\n"
            f"{'=' * 80}\n"
        )
        raise RuntimeError(error_msg)

    output_file = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False).name
    meta_data = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    with open(meta_data, "w") as fp:
        json.dump(template, fp)

    args = [
        "--inputImageList",
        label,
        "--inputDICOMDirectory",
        series_dir,
        "--outputDICOM",
        output_file,
        "--inputMetadata",
        meta_data,
    ]
    run_command(command, args)
    os.unlink(meta_data)
    return output_file


def dicom_seg_to_itk_image(label, output_ext=".seg.nrrd"):
    """Convert DICOM SEG to NIfTI/NRRD format using highdicom.

    Args:
        label: Path to DICOM SEG file or directory containing it
        output_ext: Output file extension (default: ".seg.nrrd", also supports ".nii.gz")

    Returns:
        Path to output file, or None if conversion fails
    """
    # Handle both file and directory inputs
    if os.path.isdir(label):
        # List and sort files for deterministic behavior
        files = sorted(os.listdir(label))

        # Filter for valid DICOM files
        filename = None
        for f in files:
            filepath = os.path.join(label, f)
            if not os.path.isfile(filepath):
                continue
            try:
                # Attempt to read as DICOM
                pydicom.dcmread(filepath, stop_before_pixels=True)
                filename = filepath
                break
            except (pydicom.errors.InvalidDicomError, OSError, PermissionError) as e:
                # Not a valid DICOM or inaccessible file, log and continue searching
                logger.debug(f"Skipping file {f}: {type(e).__name__}: {e}")
                continue

        if filename is None:
            raise ValueError(
                f"No valid DICOM files found in directory: {label}\n"
                f"Searched {len(files)} file(s). Ensure the directory contains valid DICOM SEG files."
            )
    else:
        filename = label

    if not HIGHDICOM_AVAILABLE:
        raise ImportError("highdicom is not available")

    # Use pydicom to read DICOM SEG
    dcm = pydicom.dcmread(filename)

    # Extract volume from DICOM SEG using highdicom
    seg_dataset = hd.seg.Segmentation.from_dataset(dcm)

    # Use get_volume() to extract the segmentation as a 3D volume
    # This automatically handles reconstruction, spacing, and geometry
    volume = seg_dataset.get_volume(combine_segments=True, relabel=True)

    # Convert to SimpleITK image
    image = SimpleITK.GetImageFromArray(volume.array)

    # Convert spacing from highdicom to SimpleITK order
    # highdicom: (slice, row, column) for axes (0, 1, 2)
    # SimpleITK: (x, y, z) = (column, row, slice)
    # Therefore: reverse the spacing tuple
    sitk_spacing = tuple(reversed(volume.spacing))
    image.SetSpacing(sitk_spacing)

    # Set origin and direction if available
    if hasattr(volume, "position") and volume.position is not None:
        # Origin (position) is in LPS physical coordinates for voxel (0,0,0)
        # Both highdicom and SimpleITK use the same coordinate system, so no conversion needed
        image.SetOrigin(volume.position)

    if hasattr(volume, "direction") and volume.direction is not None:
        # Direction matrix columns need reordering
        # highdicom: columns are [slice_dir, row_dir, col_dir] = [z, y, x]
        # SimpleITK: columns must be [x_dir, y_dir, z_dir]
        # Therefore: reorder columns from [z, y, x] to [x, y, z]
        direction_reordered = volume.direction[:, [2, 1, 0]]  # Swap columns: [0,1,2] → [2,1,0]
        direction_flat = tuple(direction_reordered.flatten())
        image.SetDirection(direction_flat)

    output_file = tempfile.NamedTemporaryFile(suffix=output_ext, delete=False).name
    SimpleITK.WriteImage(image, output_file, True)

    if not os.path.exists(output_file):
        logger.warning(f"Failed to convert DICOM-SEG {label} to ITK image")
        return None

    logger.info(f"Result/Output File: {output_file}")
    return output_file
