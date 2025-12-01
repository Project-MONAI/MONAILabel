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
from random import randint

import numpy as np
import pydicom
import SimpleITK
from monai.transforms import LoadImage
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes

try:
    import highdicom as hd
    from pydicom.sr.coding import Code

    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False
    hd = None
    Code = None

from monailabel import __version__
from monailabel.config import settings
from monailabel.datastore.utils.colors import GENERIC_ANATOMY_COLORS
from monailabel.transform.writer import write_itk

logger = logging.getLogger(__name__)


class SegmentDescription:
    """Wrapper class for segment description following MONAI Deploy pattern.

    This class encapsulates segment metadata and can convert to either:
    - highdicom.seg.SegmentDescription for the primary highdicom-based conversion
    - dcmqi JSON dict for ITK/dcmqi-based conversion (legacy fallback)
    """

    def __init__(
        self,
        segment_label,
        segmented_property_category=None,
        segmented_property_type=None,
        algorithm_name="MONAILABEL",
        algorithm_version="1.0",
        segment_description=None,
        recommended_display_rgb_value=None,
        label_id=None,
    ):
        """Initialize segment description.

        Args:
            segment_label: Label for the segment (e.g., "Spleen")
            segmented_property_category: Code for category (e.g., codes.SCT.Organ)
            segmented_property_type: Code for type (e.g., codes.SCT.Spleen)
            algorithm_name: Name of the algorithm
            algorithm_version: Version of the algorithm
            segment_description: Optional description text
            recommended_display_rgb_value: RGB color tuple [R, G, B]
            label_id: Numeric label ID
        """
        self.segment_label = segment_label
        # Use default category if not provided (safe fallback)
        if segmented_property_category is None:
            try:
                self.segmented_property_category = codes.SCT.Organ
            except Exception:
                self.segmented_property_category = None
        else:
            self.segmented_property_category = segmented_property_category
        self.segmented_property_type = segmented_property_type
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self.segment_description = segment_description or segment_label
        self.recommended_display_rgb_value = recommended_display_rgb_value or [255, 0, 0]
        self.label_id = label_id

    def to_highdicom_description(self, segment_number):
        """Convert to highdicom SegmentDescription object.

        Args:
            segment_number: Segment number (1-based)

        Returns:
            hd.seg.SegmentDescription object
        """
        if not HIGHDICOM_AVAILABLE:
            raise ImportError("highdicom is not available")

        return hd.seg.SegmentDescription(
            segment_number=segment_number,
            segment_label=self.segment_label,
            segmented_property_category=self.segmented_property_category,
            segmented_property_type=self.segmented_property_type,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name=self.algorithm_name,
                family=codes.DCM.ArtificialIntelligence,
                version=self.algorithm_version,
            ),
            algorithm_type="AUTOMATIC",
        )

    def to_dcmqi_dict(self):
        """Convert to dcmqi JSON dict for ITK-based conversion.

        Returns:
            Dictionary compatible with dcmqi itkimage2segimage
        """
        # Extract code values from pydicom Code objects
        if hasattr(self.segmented_property_type, "value"):
            type_code_value = self.segmented_property_type.value
            type_scheme = self.segmented_property_type.scheme_designator
            type_meaning = self.segmented_property_type.meaning
        else:
            type_code_value = "78961009"
            type_scheme = "SCT"
            type_meaning = self.segment_label

        return {
            "labelID": self.label_id if self.label_id is not None else 1,
            "SegmentLabel": self.segment_label,
            "SegmentDescription": self.segment_description,
            "SegmentAlgorithmType": "AUTOMATIC",
            "SegmentAlgorithmName": self.algorithm_name,
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeValue": "123037004",
                "CodingSchemeDesignator": "SCT",
                "CodeMeaning": "Anatomical Structure",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeValue": type_code_value,
                "CodingSchemeDesignator": type_scheme,
                "CodeMeaning": type_meaning,
            },
            "recommendedDisplayRGBValue": self.recommended_display_rgb_value,
        }


def random_with_n_digits(n):
    """Generate a random number with n digits."""
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def dicom_to_nifti(series_dir, is_seg=False):
    start = time.time()
    t_load = t_cpu = t_write = None

    if is_seg:
        output_file = dicom_seg_to_itk_image(series_dir)
    else:
        # Use NvDicomReader for better DICOM handling with GPU acceleration
        logger.info(f"dicom_to_nifti: Converting DICOM from {series_dir} using NvDicomReader")

        try:
            from monailabel.transform.reader import NvDicomReader

            # Use NvDicomReader with LoadImage
            reader = NvDicomReader()
            loader = LoadImage(reader=reader, image_only=False)

            # Load the DICOM (supports both directories and single files)
            t0 = time.time()
            image_data, metadata = loader(series_dir)
            t_load = time.time() - t0
            logger.info(f"dicom_to_nifti: LoadImage time: {t_load:.3f} sec")

            t1 = time.time()
            image_data = image_data.cpu().numpy()
            t_cpu = time.time() - t1
            logger.info(f"dicom_to_nifti: to.cpu().numpy() time: {t_cpu:.3f} sec")

            # Save as NIfTI using MONAI's write_itk
            output_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name

            # Get affine from metadata if available
            affine = metadata.get("affine", metadata.get("original_affine", np.eye(4)))

            t2 = time.time()
            # Use write_itk which handles the conversion properly
            write_itk(image_data, output_file, affine, image_data.dtype, compress=True)
            t_write = time.time() - t2
            logger.info(f"dicom_to_nifti: write_itk time: {t_write:.3f} sec")

        except Exception as e:
            logger.warning(f"dicom_to_nifti: NvDicomReader failed: {e}, falling back to SimpleITK")

            # Fallback to SimpleITK
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

            logger.info(f"dicom_to_nifti: Image size: {image.GetSize()}")
            output_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
            SimpleITK.WriteImage(image, output_file)

    latency = time.time() - start
    logger.info(f"dicom_to_nifti latency: {latency:.3f} sec")
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


def nifti_to_dicom_seg(
    series_dir, label, label_info, file_ext="*", use_itk=None, omit_empty_frames=False, custom_tags=None
) -> str:
    """Convert NIfTI segmentation to DICOM SEG format using highdicom or ITK (fallback).

    This function uses highdicom by default for creating DICOM SEG objects.
    The ITK/dcmqi method is available as a fallback option (use_itk=True).

    Args:
        series_dir: Directory containing source DICOM images
        label: Path to NIfTI label file
        label_info: List of dictionaries containing segment information
        file_ext: File extension pattern for DICOM files (default: "*")
        use_itk: If True, use ITK/dcmqi-based conversion (fallback). If False or None, use highdicom (default).
        omit_empty_frames: If True, omit frames with no segmented pixels (default: False to match legacy behavior)
        custom_tags: Optional dictionary of custom DICOM tags to add (keyword: value)

    Returns:
        Path to output DICOM SEG file
    """
    # Only use config if no explicit override
    if use_itk is None:
        use_itk = settings.MONAI_LABEL_USE_ITK_FOR_DICOM_SEG

    start = time.time()

    # Check if highdicom is available (unless using ITK fallback)
    if not use_itk and not HIGHDICOM_AVAILABLE:
        logger.warning("highdicom not available, falling back to ITK method")
        use_itk = True

    # Load label and get unique segments
    label_np, meta_dict = LoadImage(image_only=False)(label)
    unique_labels = np.unique(label_np.flatten()).astype(np.int_)
    unique_labels = unique_labels[unique_labels != 0]

    info = label_info[0] if label_info and 0 < len(label_info) else {}
    model_name = info.get("model_name", "AIName")

    if not unique_labels.size:
        logger.error("No non-zero labels found in segmentation")
        return ""

    # Build segment descriptions
    segment_descriptions = []
    for i, idx in enumerate(unique_labels):
        info = label_info[i] if label_info and i < len(label_info) else {}
        name = info.get("name", f"Segment_{idx}")
        description = info.get("description", name)

        logger.info(f"Segment {i}: idx={idx}, name={name}")

        if use_itk:
            # Build template for ITK method
            rgb = list(info.get("color", GENERIC_ANATOMY_COLORS.get(name, (255, 0, 0))))[0:3]
            rgb = [int(x) for x in rgb]

            segment_attr = info.get(
                "segmentAttribute",
                {
                    "labelID": int(idx),
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
                },
            )
            segment_descriptions.append(segment_attr)
        else:
            # Build highdicom SegmentDescription
            # Get codes from label_info or use defaults
            category_code = codes.SCT.Organ  # Default: Organ
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
            seg_desc = hd.seg.SegmentDescription(
                segment_number=int(idx),
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

    if use_itk:
        # Use ITK method
        template = {
            "ContentCreatorName": "Reader1",
            "ClinicalTrialSeriesID": "Session1",
            "ClinicalTrialTimePointID": "1",
            "SeriesDescription": model_name,
            "SeriesNumber": "300",
            "InstanceNumber": "1",
            "segmentAttributes": [segment_descriptions],
            "ContentLabel": "SEGMENTATION",
            "ContentDescription": "MONAI Label - Image segmentation",
            "ClinicalTrialCoordinatingCenterName": "MONAI",
            "BodyPartExamined": "",
        }
        logger.info(json.dumps(template, indent=2))
        output_file = itk_image_to_dicom_seg(label, series_dir, template)
    else:
        # Use highdicom method
        # Read source DICOM images (headers only for memory efficiency)
        series_dir = pathlib.Path(series_dir)
        image_files = list(series_dir.glob(file_ext))
        image_datasets = [dcmread(str(f), stop_before_pixels=True) for f in sorted(image_files)]
        logger.info(f"Total Source Images: {len(image_datasets)}")

        if not image_datasets:
            logger.error(f"No DICOM images found in {series_dir} with pattern {file_ext}")
            return ""

        # Load label using SimpleITK and convert to numpy array
        # Use uint16 to support up to 65,535 segments
        mask = SimpleITK.ReadImage(label)
        mask = SimpleITK.Cast(mask, SimpleITK.sitkUInt16)

        # Convert to numpy array for highdicom
        seg_array = SimpleITK.GetArrayFromImage(mask)

        # Remap label values to sequential 1, 2, 3... as required by highdicom
        # (highdicom requires explicit sequential remapping)
        remapped_array = np.zeros_like(seg_array, dtype=np.uint16)
        for new_idx, orig_idx in enumerate(unique_labels, start=1):
            remapped_array[seg_array == orig_idx] = new_idx
        seg_array = remapped_array

        # Generate SOP instance UID
        seg_sop_instance_uid = hd.UID()

        # Create DICOM SEG using highdicom
        try:
            # Get software version
            try:
                software_version = f"MONAI Label {__version__}"
            except Exception:
                software_version = "MONAI Label"

            seg = hd.seg.Segmentation(
                source_images=image_datasets,
                pixel_array=seg_array,
                segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                segment_descriptions=segment_descriptions,
                series_instance_uid=hd.UID(),
                series_number=random_with_n_digits(4),
                sop_instance_uid=seg_sop_instance_uid,
                instance_number=1,
                manufacturer="MONAI Consortium",
                manufacturer_model_name="MONAI Label",
                software_versions=software_version,
                device_serial_number="0000",
                omit_empty_frames=omit_empty_frames,
            )

            # Add timestamp and timezone
            dt_now = datetime.datetime.now()
            seg.SeriesDate = dt_now.strftime("%Y%m%d")
            seg.SeriesTime = dt_now.strftime("%H%M%S")
            seg.TimezoneOffsetFromUTC = dt_now.astimezone().isoformat()[-6:].replace(":", "")  # Format: +0000 or -0700
            seg.SeriesDescription = model_name

            # Add Contributing Equipment Sequence (following MONAI Deploy pattern)
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
            except Exception as e:
                logger.warning(f"Could not add ContributingEquipmentSequence: {e}")

            # Add custom tags if provided (following MONAI Deploy pattern)
            if custom_tags:
                for k, v in custom_tags.items():
                    if isinstance(k, str) and isinstance(v, str):
                        try:
                            if k in seg:
                                data_element = seg.data_element(k)
                                if data_element:
                                    data_element.value = v
                            else:
                                seg.update({k: v})
                        except Exception as ex:
                            logger.warning(f"Custom tag {k} was not written, due to {ex}")

            # Save DICOM SEG
            output_file = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False).name
            seg.save_as(output_file)
            logger.info(f"DICOM SEG saved to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to create DICOM SEG with highdicom: {e}")
            logger.info("Falling back to ITK method")
            # Fallback to ITK method
            template = {
                "ContentCreatorName": "Reader1",
                "SeriesDescription": model_name,
                "SeriesNumber": "300",
                "InstanceNumber": "1",
                "segmentAttributes": [
                    [
                        {
                            "labelID": int(idx),
                            "SegmentLabel": info.get("name", f"Segment_{idx}"),
                            "SegmentDescription": info.get("description", ""),
                            "SegmentAlgorithmType": "AUTOMATIC",
                            "SegmentAlgorithmName": "MONAILABEL",
                        }
                        for idx, info in zip(unique_labels, label_info or [])
                    ]
                ],
                "ContentLabel": "SEGMENTATION",
                "ContentDescription": "MONAI Label - Image segmentation",
            }
            output_file = itk_image_to_dicom_seg(label, str(series_dir), template)

    logger.info(f"nifti_to_dicom_seg latency: {time.time() - start:.3f} sec")
    return output_file


def itk_image_to_dicom_seg(label, series_dir, template) -> str:
    import shutil

    from monailabel.utils.others.generic import run_command

    command = "itkimage2segimage"
    if not shutil.which(command):
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: {command} command-line tool not found\n"
            f"{'='*80}\n\n"
            f"The ITK-based DICOM SEG conversion requires the dcmqi package.\n\n"
            f"Install dcmqi:\n"
            f"  pip install dcmqi\n\n"
            f"For more information:\n"
            f"  https://github.com/QIICR/dcmqi\n\n"
            f"Note: Consider using the default highdicom-based conversion (use_itk=False)\n"
            f"which doesn't require dcmqi.\n"
            f"{'='*80}\n"
        )
        raise RuntimeError(error_msg)

    output_file = tempfile.NamedTemporaryFile(suffix=".dcm").name
    meta_data = tempfile.NamedTemporaryFile(suffix=".json").name
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
    """Convert DICOM SEG to ITK image format using highdicom.

    Args:
        label: Path to DICOM SEG file or directory containing it
        output_ext: Output file extension (default: ".seg.nrrd")

    Returns:
        Path to output file, or None if conversion fails
    """
    filename = label if not os.path.isdir(label) else os.path.join(label, os.listdir(label)[0])

    if not HIGHDICOM_AVAILABLE:
        raise ImportError("highdicom is not available")

    # Use pydicom to read DICOM SEG
    dcm = pydicom.dcmread(filename)

    # Extract pixel array from DICOM SEG
    seg_dataset = hd.seg.Segmentation.from_dataset(dcm)
    pixel_array = seg_dataset.get_total_pixel_matrix()

    # Convert to SimpleITK image
    image = SimpleITK.GetImageFromArray(pixel_array)

    # Try to get spacing and other metadata from original DICOM
    if hasattr(dcm, "SharedFunctionalGroupsSequence") and len(dcm.SharedFunctionalGroupsSequence) > 0:
        shared_func_groups = dcm.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_func_groups, "PixelMeasuresSequence"):
            pixel_measures = shared_func_groups.PixelMeasuresSequence[0]
            if hasattr(pixel_measures, "PixelSpacing"):
                spacing = list(pixel_measures.PixelSpacing)
                if hasattr(pixel_measures, "SliceThickness"):
                    spacing.append(float(pixel_measures.SliceThickness))
                image.SetSpacing(spacing)

    output_file = tempfile.NamedTemporaryFile(suffix=output_ext, delete=False).name
    SimpleITK.WriteImage(image, output_file, True)

    if not os.path.exists(output_file):
        logger.warning(f"Failed to convert DICOM-SEG {label} to ITK image")
        return None

    logger.info(f"Result/Output File: {output_file}")
    return output_file
