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

# Global singleton instances for nvimgcodec encoder/decoder
# These are initialized lazily on first use to avoid import errors
# when nvimgcodec is not available
_NVIMGCODEC_ENCODER = None
_NVIMGCODEC_DECODER = None


def _get_nvimgcodec_encoder():
    """Get or create the global nvimgcodec encoder singleton."""
    global _NVIMGCODEC_ENCODER
    if _NVIMGCODEC_ENCODER is None:
        try:
            from nvidia import nvimgcodec
            _NVIMGCODEC_ENCODER = nvimgcodec.Encoder()
            logger.debug("Initialized global nvimgcodec.Encoder singleton")
        except ImportError:
            raise ImportError(
                "nvidia-nvimgcodec is required for HTJ2K transcoding. "
                "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
                "(replace {XX} with your CUDA version, e.g., cu13)"
            )
    return _NVIMGCODEC_ENCODER


def _get_nvimgcodec_decoder():
    """Get or create the global nvimgcodec decoder singleton."""
    global _NVIMGCODEC_DECODER
    if _NVIMGCODEC_DECODER is None:
        try:
            from nvidia import nvimgcodec
            _NVIMGCODEC_DECODER = nvimgcodec.Decoder()
            logger.debug("Initialized global nvimgcodec.Decoder singleton")
        except ImportError:
            raise ImportError(
                "nvidia-nvimgcodec is required for HTJ2K decoding. "
                "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
                "(replace {XX} with your CUDA version, e.g., cu13)"
            )
    return _NVIMGCODEC_DECODER


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
    from monailabel.utils.others.generic import run_command
    import shutil

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


def _setup_htj2k_decode_params():
    """
    Create nvimgcodec decoding parameters for DICOM images.
    
    Returns:
        nvimgcodec.DecodeParams: Decode parameters configured for DICOM
    """
    from nvidia import nvimgcodec
    
    decode_params = nvimgcodec.DecodeParams(
        allow_any_depth=True,
        color_spec=nvimgcodec.ColorSpec.UNCHANGED,
    )
    
    return decode_params


def _setup_htj2k_encode_params(num_resolutions: int = 6, code_block_size: tuple = (64, 64)):
    """
    Create nvimgcodec encoding parameters for HTJ2K lossless compression.
    
    Args:
        num_resolutions: Number of wavelet decomposition levels
        code_block_size: Code block size as (height, width) tuple
        
    Returns:
        tuple: (encode_params, target_transfer_syntax)
    """
    from nvidia import nvimgcodec
    
    target_transfer_syntax = "1.2.840.10008.1.2.4.202"  # HTJ2K with RPCL Options (Lossless)
    quality_type = nvimgcodec.QualityType.LOSSLESS
    
    # Configure JPEG2K encoding parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params.num_resolutions = num_resolutions
    jpeg2k_encode_params.code_block_size = code_block_size
    jpeg2k_encode_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2
    jpeg2k_encode_params.prog_order = nvimgcodec.Jpeg2kProgOrder.LRCP
    jpeg2k_encode_params.ht = True  # Enable High Throughput mode
    
    encode_params = nvimgcodec.EncodeParams(
        quality_type=quality_type,
        jpeg2k_encode_params=jpeg2k_encode_params,
    )
    
    return encode_params, target_transfer_syntax


def _get_transfer_syntax_constants():
    """
    Get transfer syntax UID constants for categorizing DICOM files.
    
    Returns:
        dict: Dictionary with keys 'JPEG2000', 'HTJ2K', 'JPEG', 'NVIMGCODEC' (combined set)
    """
    JPEG2000_SYNTAXES = frozenset([
        "1.2.840.10008.1.2.4.90",  # JPEG 2000 Image Compression (Lossless Only)
        "1.2.840.10008.1.2.4.91",  # JPEG 2000 Image Compression
    ])
    
    HTJ2K_SYNTAXES = frozenset([
        "1.2.840.10008.1.2.4.201",  # High-Throughput JPEG 2000 Image Compression (Lossless Only)
        "1.2.840.10008.1.2.4.202",  # High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)
        "1.2.840.10008.1.2.4.203",  # High-Throughput JPEG 2000 Image Compression
    ])
    
    JPEG_SYNTAXES = frozenset([
        "1.2.840.10008.1.2.4.50",  # JPEG Baseline (Process 1)
        "1.2.840.10008.1.2.4.51",  # JPEG Extended (Process 2 & 4)
        "1.2.840.10008.1.2.4.57",  # JPEG Lossless, Non-Hierarchical (Process 14)
        "1.2.840.10008.1.2.4.70",  # JPEG Lossless, Non-Hierarchical, First-Order Prediction
    ])
    
    return {
        'JPEG2000': JPEG2000_SYNTAXES,
        'HTJ2K': HTJ2K_SYNTAXES,
        'JPEG': JPEG_SYNTAXES,
        'NVIMGCODEC': JPEG2000_SYNTAXES | HTJ2K_SYNTAXES | JPEG_SYNTAXES
    }


def _create_basic_offset_table_pixel_data(encoded_frames: list) -> bytes:
    """
    Create encapsulated pixel data with Basic Offset Table for multi-frame DICOM.
    
    Uses pydicom's encapsulate() function to ensure 100% standard compliance.
    
    Args:
        encoded_frames: List of encoded frame byte strings
        
    Returns:
        bytes: Encapsulated pixel data with Basic Offset Table per DICOM Part 5 Section A.4
    """
    return pydicom.encaps.encapsulate(encoded_frames, has_bot=True)


def transcode_dicom_to_htj2k(
    input_dir: str,
    output_dir: str = None,
    num_resolutions: int = 6,
    code_block_size: tuple = (64, 64),
    max_batch_size: int = 256,
    add_basic_offset_table: bool = True,
) -> str:
    """
    Transcode DICOM files to HTJ2K (High Throughput JPEG 2000) lossless compression.
    
    HTJ2K is a faster variant of JPEG 2000 that provides better compression performance
    for medical imaging applications. This function uses nvidia-nvimgcodec for hardware-
    accelerated decoding and encoding with batch processing for optimal performance.
    All transcoding is performed using lossless compression to preserve image quality.
    
    The function processes files in configurable batches:
    1. Categorizes files by transfer syntax (HTJ2K/JPEG2000/JPEG/uncompressed)
    2. Uses nvimgcodec decoder for compressed files (HTJ2K, JPEG2000, JPEG)
    3. Falls back to pydicom pixel_array for uncompressed files
    4. Batch encodes all images to HTJ2K using nvimgcodec
    5. Saves transcoded files with updated transfer syntax and optional Basic Offset Table
    
    Supported source transfer syntaxes:
    - HTJ2K (High-Throughput JPEG 2000) - decoded and re-encoded to add BOT if needed
    - JPEG 2000 (lossless and lossy)
    - JPEG (baseline, extended, lossless)
    - Uncompressed (Explicit/Implicit VR Little/Big Endian)

    Typical compression ratios of 60-70% with lossless quality.
    Processing speed depends on batch size and GPU capabilities.
    
    Args:
        input_dir: Path to directory containing DICOM files to transcode
        output_dir: Path to output directory for transcoded files. If None, creates temp directory
        num_resolutions: Number of wavelet decomposition levels (default: 6)
                        Higher values = better compression but slower encoding
        code_block_size: Code block size as (height, width) tuple (default: (64, 64))
                        Must be powers of 2. Common values: (32,32), (64,64), (128,128)
        max_batch_size: Maximum number of DICOM files to process in each batch (default: 256)
                       Lower values reduce memory usage, higher values may improve speed
        add_basic_offset_table: If True, creates Basic Offset Table for multi-frame DICOMs (default: True)
                               BOT enables O(1) frame access without parsing entire pixel data stream
                               Per DICOM Part 5 Section A.4. Only affects multi-frame files.
        
    Returns:
        str: Path to output directory containing transcoded DICOM files
        
    Raises:
        ImportError: If nvidia-nvimgcodec is not available
        ValueError: If input directory doesn't exist or contains no valid DICOM files
        ValueError: If DICOM files are missing required attributes (TransferSyntaxUID, PixelData)
        
    Example:
        >>> # Basic usage with default settings
        >>> output_dir = transcode_dicom_to_htj2k("/path/to/dicoms")
        >>> print(f"Transcoded files saved to: {output_dir}")
        
        >>> # Custom output directory and batch size
        >>> output_dir = transcode_dicom_to_htj2k(
        ...     input_dir="/path/to/dicoms",
        ...     output_dir="/path/to/output",
        ...     max_batch_size=50,
        ...     num_resolutions=5
        ... )
        
        >>> # Process with smaller code blocks for memory efficiency
        >>> output_dir = transcode_dicom_to_htj2k(
        ...     input_dir="/path/to/dicoms",
        ...     code_block_size=(32, 32),
        ...     max_batch_size=5
        ... )
        
    Note:
        Requires nvidia-nvimgcodec to be installed:
            pip install nvidia-nvimgcodec-cu{XX}[all]
        Replace {XX} with your CUDA version (e.g., cu13 for CUDA 13.x)
        
        The function preserves all DICOM metadata including Patient, Study, and Series
        information. Only the transfer syntax and pixel data encoding are modified.
    """
    import glob
    import shutil
    from pathlib import Path
    
    # Check for nvidia-nvimgcodec
    try:
        from nvidia import nvimgcodec
    except ImportError:
        raise ImportError(
            "nvidia-nvimgcodec is required for HTJ2K transcoding. "
            "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
            "(replace {XX} with your CUDA version, e.g., cu13)"
        )
    
    # Validate input
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Get all DICOM files
    dicom_files = []
    for pattern in ["*.dcm", "*"]:
        dicom_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    # Filter to actual DICOM files
    valid_dicom_files = []
    for file_path in dicom_files:
        if os.path.isfile(file_path):
            try:
                # Quick check if it's a DICOM file
                with open(file_path, 'rb') as f:
                    f.seek(128)
                    magic = f.read(4)
                    if magic == b'DICM':
                        valid_dicom_files.append(file_path)
            except Exception:
                continue
    
    if not valid_dicom_files:
        raise ValueError(f"No valid DICOM files found in {input_dir}")
    
    logger.info(f"Found {len(valid_dicom_files)} DICOM files to transcode")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="htj2k_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create encoder and decoder instances (reused for all files)
    encoder = _get_nvimgcodec_encoder()
    decoder = _get_nvimgcodec_decoder()  # Always needed for decoding input DICOM images
    
    # Setup HTJ2K encoding and decoding parameters
    encode_params, target_transfer_syntax = _setup_htj2k_encode_params(
        num_resolutions=num_resolutions,
        code_block_size=code_block_size
    )
    decode_params = _setup_htj2k_decode_params()
    logger.info("Using lossless HTJ2K compression")
    
    # Get transfer syntax constants
    ts_constants = _get_transfer_syntax_constants()
    NVIMGCODEC_SYNTAXES = ts_constants['NVIMGCODEC']
    
    start_time = time.time()
    transcoded_count = 0
    
    # Calculate batch info for logging
    total_files = len(valid_dicom_files)
    total_batches = (total_files + max_batch_size - 1) // max_batch_size
    
    for batch_start in range(0, total_files, max_batch_size):
        batch_end = min(batch_start + max_batch_size, total_files)
        current_batch = batch_start // max_batch_size + 1
        logger.info(f"[{batch_start}..{batch_end}] Processing batch {current_batch}/{total_batches}")
        batch_files = valid_dicom_files[batch_start:batch_end]
        batch_datasets = [pydicom.dcmread(file) for file in batch_files]
        nvimgcodec_batch = []
        pydicom_batch = []
        
        for idx, ds in enumerate(batch_datasets):
            current_ts = getattr(ds, 'file_meta', {}).get('TransferSyntaxUID', None)
            if current_ts is None:
                raise ValueError(f"DICOM file {os.path.basename(batch_files[idx])} does not have a Transfer Syntax UID")
            
            ts_str = str(current_ts)
            if ts_str in NVIMGCODEC_SYNTAXES:
                if not hasattr(ds, "PixelData") or ds.PixelData is None:
                    raise ValueError(f"DICOM file {os.path.basename(batch_files[idx])} does not have a PixelData member")
                nvimgcodec_batch.append(idx)

            else:
                pydicom_batch.append(idx)
        
        data_sequence = []
        decoded_data = []
        num_frames = []
        
        # Decode using nvimgcodec for compressed formats
        if nvimgcodec_batch:
            for idx in nvimgcodec_batch:
                frames = [fragment for fragment in pydicom.encaps.generate_frames(batch_datasets[idx].PixelData)]
                num_frames.append(len(frames))
                data_sequence.extend(frames)
            decoder_output = decoder.decode(data_sequence, params=decode_params)
            decoded_data.extend(decoder_output)

        # Decode using pydicom for uncompressed formats
        if pydicom_batch:
            for idx in pydicom_batch:
                source_pixel_array = batch_datasets[idx].pixel_array
                if not isinstance(source_pixel_array, np.ndarray):
                    source_pixel_array = np.array(source_pixel_array)
                if source_pixel_array.ndim == 2:
                    source_pixel_array = source_pixel_array[:, :, np.newaxis]
                for frame_idx in range(source_pixel_array.shape[-1]):
                    decoded_data.append(source_pixel_array[:, :, frame_idx])
                num_frames.append(source_pixel_array.shape[-1])

        # Encode all frames to HTJ2K
        encoded_data = encoder.encode(decoded_data, codec="jpeg2k", params=encode_params)

        # Reassemble and save transcoded files
        frame_offset = 0
        files_to_process = nvimgcodec_batch + pydicom_batch
        
        for list_idx, dataset_idx in enumerate(files_to_process):
            nframes = num_frames[list_idx]
            encoded_frames = [bytes(enc) for enc in encoded_data[frame_offset:frame_offset + nframes]]
            frame_offset += nframes
            
            # Update dataset with HTJ2K encoded data
            # Create Basic Offset Table for multi-frame files if requested
            if add_basic_offset_table and nframes > 1:
                batch_datasets[dataset_idx].PixelData = _create_basic_offset_table_pixel_data(encoded_frames)
                logger.debug(f"Created Basic Offset Table for {os.path.basename(batch_files[dataset_idx])} ({nframes} frames)")
            else:
                batch_datasets[dataset_idx].PixelData = pydicom.encaps.encapsulate(encoded_frames)

            batch_datasets[dataset_idx].file_meta.TransferSyntaxUID = pydicom.uid.UID(target_transfer_syntax)
            
            # Save transcoded file
            output_file = os.path.join(output_dir, os.path.basename(batch_files[dataset_idx]))
            batch_datasets[dataset_idx].save_as(output_file)
            transcoded_count += 1
    
    elapsed_time = time.time() - start_time

    logger.info(f"Transcoding complete:")
    logger.info(f"  Total files: {len(valid_dicom_files)}")
    logger.info(f"  Successfully transcoded: {transcoded_count}")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"  Output directory: {output_dir}")
    
    return output_dir


def transcode_dicom_to_htj2k_multiframe(
    input_dir: str,
    output_dir: str = None,
    num_resolutions: int = 6,
    code_block_size: tuple = (64, 64),
) -> str:
    """
    Transcode DICOM files to HTJ2K and combine all frames from the same series into single multi-frame files.
    
    This function groups DICOM files by SeriesInstanceUID and combines all frames from each series
    into a single multi-frame DICOM file with HTJ2K compression. This is useful for:
    - Reducing file count (one file per series instead of many)
    - Improving storage efficiency
    - Enabling more efficient frame-level access patterns
    
    The function:
    1. Scans input directory recursively for DICOM files
    2. Groups files by StudyInstanceUID and SeriesInstanceUID
    3. For each series, decodes all frames and combines them
    4. Encodes combined frames to HTJ2K
    5. Creates a Basic Offset Table for efficient frame access (per DICOM Part 5 Section A.4)
    6. Saves as a single multi-frame DICOM file per series
    
    Args:
        input_dir: Path to directory containing DICOM files (will scan recursively)
        output_dir: Path to output directory for transcoded files. If None, creates temp directory
        num_resolutions: Number of wavelet decomposition levels (default: 6)
        code_block_size: Code block size as (height, width) tuple (default: (64, 64))
        
    Returns:
        str: Path to output directory containing transcoded multi-frame DICOM files
        
    Raises:
        ImportError: If nvidia-nvimgcodec is not available
        ValueError: If input directory doesn't exist or contains no valid DICOM files
        
    Example:
        >>> # Combine series and transcode to HTJ2K
        >>> output_dir = transcode_dicom_to_htj2k_multiframe("/path/to/dicoms")
        >>> print(f"Multi-frame files saved to: {output_dir}")
        
    Note:
        Each output file is named using the SeriesInstanceUID:
        <StudyUID>/<SeriesUID>.dcm
        
        The NumberOfFrames tag is set to the total frame count.
        All other DICOM metadata is preserved from the first instance in each series.
        
        Basic Offset Table:
        A Basic Offset Table is automatically created containing byte offsets to each frame.
        This allows DICOM readers to quickly locate and extract individual frames without
        parsing the entire encapsulated pixel data stream. The offsets are 32-bit unsigned
        integers measured from the first byte of the first Item Tag following the BOT.
    """
    import glob
    import shutil
    import tempfile
    from collections import defaultdict
    from pathlib import Path
    
    # Check for nvidia-nvimgcodec
    try:
        from nvidia import nvimgcodec
    except ImportError:
        raise ImportError(
            "nvidia-nvimgcodec is required for HTJ2K transcoding. "
            "Install it with: pip install nvidia-nvimgcodec-cu{XX}[all] "
            "(replace {XX} with your CUDA version, e.g., cu13)"
        )
    
    import pydicom
    import numpy as np
    import time
    
    # Validate input
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Get all DICOM files recursively
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dcm') or file.endswith('.DCM'):
                dicom_files.append(os.path.join(root, file))
    
    # Also check for files without extension
    for pattern in ["*"]:
        found_files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
        for file_path in found_files:
            if os.path.isfile(file_path) and file_path not in dicom_files:
                try:
                    with open(file_path, 'rb') as f:
                        f.seek(128)
                        magic = f.read(4)
                        if magic == b'DICM':
                            dicom_files.append(file_path)
                except Exception:
                    continue
    
    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in {input_dir}")
    
    logger.info(f"Found {len(dicom_files)} DICOM files to process")
    
    # Group files by study and series
    series_groups = defaultdict(list)  # Key: (StudyUID, SeriesUID), Value: list of file paths
    
    logger.info("Grouping DICOM files by series...")
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            study_uid = str(ds.StudyInstanceUID)
            series_uid = str(ds.SeriesInstanceUID)
            instance_number = int(getattr(ds, 'InstanceNumber', 0))
            series_groups[(study_uid, series_uid)].append((instance_number, file_path))
        except Exception as e:
            logger.warning(f"Failed to read metadata from {file_path}: {e}")
            continue
    
    # Sort files within each series by InstanceNumber
    for key in series_groups:
        series_groups[key].sort(key=lambda x: x[0])  # Sort by instance number
    
    logger.info(f"Found {len(series_groups)} unique series")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create encoder and decoder instances
    encoder = _get_nvimgcodec_encoder()
    decoder = _get_nvimgcodec_decoder()
    
    # Setup HTJ2K encoding and decoding parameters
    encode_params, target_transfer_syntax = _setup_htj2k_encode_params(
        num_resolutions=num_resolutions,
        code_block_size=code_block_size
    )
    decode_params = _setup_htj2k_decode_params()
    
    # Get transfer syntax constants
    ts_constants = _get_transfer_syntax_constants()
    NVIMGCODEC_SYNTAXES = ts_constants['NVIMGCODEC']
    
    start_time = time.time()
    processed_series = 0
    total_frames = 0
    
    # Process each series
    for (study_uid, series_uid), file_list in series_groups.items():
        try:
            logger.info(f"Processing series {series_uid} ({len(file_list)} instances)")
            
            # Load all datasets for this series
            file_paths = [fp for _, fp in file_list]
            datasets = [pydicom.dcmread(fp) for fp in file_paths]
            
            # CRITICAL: Sort datasets by ImagePositionPatient Z-coordinate
            # This ensures Frame[0] is the first slice, Frame[N] is the last slice
            if all(hasattr(ds, 'ImagePositionPatient') for ds in datasets):
                # Sort by Z coordinate (3rd element of ImagePositionPatient)
                datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
                logger.info(f"  ✓ Sorted {len(datasets)} frames by ImagePositionPatient Z-coordinate")
                logger.info(f"    First frame Z: {datasets[0].ImagePositionPatient[2]}")
                logger.info(f"    Last frame Z:  {datasets[-1].ImagePositionPatient[2]}")
                
                # NOTE: We keep anatomically correct order (Z-ascending)
                # Cornerstone3D should use per-frame ImagePositionPatient from PerFrameFunctionalGroupsSequence
                # We provide complete per-frame metadata (PlanePositionSequence + PlaneOrientationSequence)
                logger.info(f"  ✓ Frames in anatomical order (lowest Z first)")
                logger.info(f"    Cornerstone3D should use per-frame ImagePositionPatient for correct volume reconstruction")
            else:
                logger.warning(f"  ⚠️  Some frames missing ImagePositionPatient, using file order")
            
            # Use first dataset as template
            template_ds = datasets[0]
            
            # Collect all frames from all instances
            all_decoded_frames = []
            
            for ds in datasets:
                current_ts = str(getattr(ds.file_meta, 'TransferSyntaxUID', None))
                
                if current_ts in NVIMGCODEC_SYNTAXES:
                    # Compressed format - use nvimgcodec decoder
                    frames = [fragment for fragment in pydicom.encaps.generate_frames(ds.PixelData)]
                    decoded = decoder.decode(frames, params=decode_params)
                    all_decoded_frames.extend(decoded)
                else:
                    # Uncompressed format - use pydicom
                    pixel_array = ds.pixel_array
                    if not isinstance(pixel_array, np.ndarray):
                        pixel_array = np.array(pixel_array)
                    
                    # Handle single frame vs multi-frame
                    if pixel_array.ndim == 2:
                        # Single frame
                        pixel_array = pixel_array[:, :, np.newaxis]
                        all_decoded_frames.append(pixel_array)
                    elif pixel_array.ndim == 3:
                        # Multi-frame (frames are first dimension)
                        for frame_idx in range(pixel_array.shape[0]):
                            frame_2d = pixel_array[frame_idx, :, :]
                            if frame_2d.ndim == 2:
                                frame_2d = frame_2d[:, :, np.newaxis]
                            all_decoded_frames.append(frame_2d)
            
            total_frame_count = len(all_decoded_frames)
            logger.info(f"  Total frames in series: {total_frame_count}")
            
            # Encode all frames to HTJ2K
            logger.info(f"  Encoding {total_frame_count} frames to HTJ2K...")
            encoded_frames = encoder.encode(all_decoded_frames, codec="jpeg2k", params=encode_params)
            
            # Convert to bytes
            encoded_frames_bytes = [bytes(enc) for enc in encoded_frames]
            
            # Create SIMPLE multi-frame DICOM file (like the user's example)
            # Use first dataset as template, keeping its metadata
            logger.info(f"  Creating simple multi-frame DICOM from {total_frame_count} frames...")
            output_ds = datasets[0].copy()  # Start from first dataset
            
            # Update pixel data with all HTJ2K encoded frames + Basic Offset Table
            output_ds.PixelData = _create_basic_offset_table_pixel_data(encoded_frames_bytes)
            output_ds.file_meta.TransferSyntaxUID = pydicom.uid.UID(target_transfer_syntax)
            
            # Set NumberOfFrames (critical!)
            output_ds.NumberOfFrames = total_frame_count
            
            # DICOM Multi-frame Module (C.7.6.6) - Mandatory attributes
            
            # FrameIncrementPointer - REQUIRED to tell viewers how frames are ordered
            # Points to ImagePositionPatient (0020,0032) which varies per frame
            output_ds.FrameIncrementPointer = 0x00200032
            logger.info(f"  ✓ Set FrameIncrementPointer to ImagePositionPatient")
            
            # Ensure all Image Pixel Module attributes are present (C.7.6.3)
            # These should be inherited from first frame, but verify:
            required_pixel_attrs = [
                ('SamplesPerPixel', 1),
                ('PhotometricInterpretation', 'MONOCHROME2'),
                ('Rows', 512),
                ('Columns', 512),
            ]
            
            for attr, default in required_pixel_attrs:
                if not hasattr(output_ds, attr):
                    setattr(output_ds, attr, default)
                    logger.warning(f"  ⚠️  Added missing {attr} = {default}")
            
            # Keep first frame's spatial attributes as top-level (represents volume origin)
            if hasattr(datasets[0], 'ImagePositionPatient'):
                output_ds.ImagePositionPatient = datasets[0].ImagePositionPatient
                logger.info(f"  ✓ Top-level ImagePositionPatient: {output_ds.ImagePositionPatient}")
                logger.info(f"    (This is Frame[0], the FIRST slice in Z-order)")
            
            if hasattr(datasets[0], 'ImageOrientationPatient'):
                output_ds.ImageOrientationPatient = datasets[0].ImageOrientationPatient
                logger.info(f"  ✓ ImageOrientationPatient: {output_ds.ImageOrientationPatient}")
            
            # Keep pixel spacing and slice thickness
            if hasattr(datasets[0], 'PixelSpacing'):
                output_ds.PixelSpacing = datasets[0].PixelSpacing
                logger.info(f"  ✓ PixelSpacing: {output_ds.PixelSpacing}")
            
            if hasattr(datasets[0], 'SliceThickness'):
                output_ds.SliceThickness = datasets[0].SliceThickness
                logger.info(f"  ✓ SliceThickness: {output_ds.SliceThickness}")
            
            # Fix InstanceNumber (should be >= 1)
            output_ds.InstanceNumber = 1
            
            # Ensure SeriesNumber is present
            if not hasattr(output_ds, 'SeriesNumber'):
                output_ds.SeriesNumber = 1
            
            # Remove per-frame tags that conflict with multi-frame
            if hasattr(output_ds, 'SliceLocation'):
                delattr(output_ds, 'SliceLocation')
                logger.info(f"  ✓ Removed SliceLocation (per-frame tag)")
            
            # Add SpacingBetweenSlices
            if len(datasets) > 1:
                pos0 = datasets[0].ImagePositionPatient if hasattr(datasets[0], 'ImagePositionPatient') else None
                pos1 = datasets[1].ImagePositionPatient if hasattr(datasets[1], 'ImagePositionPatient') else None
                
                if pos0 and pos1:
                    # Calculate spacing as distance between consecutive slices
                    import math
                    spacing = math.sqrt(sum((float(pos1[i]) - float(pos0[i]))**2 for i in range(3)))
                    output_ds.SpacingBetweenSlices = spacing
                    logger.info(f"  ✓ Added SpacingBetweenSlices: {spacing:.6f} mm")
            
            # Add minimal PerFrameFunctionalGroupsSequence for OHIF compatibility
            # OHIF's cornerstone3D expects this even for simple multi-frame CT
            logger.info(f"  Adding minimal per-frame functional groups for OHIF compatibility...")
            from pydicom.sequence import Sequence
            from pydicom.dataset import Dataset as DicomDataset
            
            per_frame_seq = []
            for frame_idx, ds_frame in enumerate(datasets):
                frame_item = DicomDataset()
                
                # PlanePositionSequence - ImagePositionPatient for this frame
                # CRITICAL: Best defense against Cornerstone3D bugs
                if hasattr(ds_frame, 'ImagePositionPatient'):
                    plane_pos_item = DicomDataset()
                    plane_pos_item.ImagePositionPatient = ds_frame.ImagePositionPatient
                    frame_item.PlanePositionSequence = Sequence([plane_pos_item])
                
                # PlaneOrientationSequence - ImageOrientationPatient for this frame
                # CRITICAL: Best defense against Cornerstone3D bugs
                if hasattr(ds_frame, 'ImageOrientationPatient'):
                    plane_orient_item = DicomDataset()
                    plane_orient_item.ImageOrientationPatient = ds_frame.ImageOrientationPatient
                    frame_item.PlaneOrientationSequence = Sequence([plane_orient_item])
                
                # FrameContentSequence - helps with frame identification
                frame_content_item = DicomDataset()
                frame_content_item.StackID = "1"
                frame_content_item.InStackPositionNumber = frame_idx + 1
                frame_content_item.DimensionIndexValues = [1, frame_idx + 1]
                frame_item.FrameContentSequence = Sequence([frame_content_item])
                
                per_frame_seq.append(frame_item)
            
            output_ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_seq)
            logger.info(f"  ✓ Added PerFrameFunctionalGroupsSequence with {len(per_frame_seq)} frame items")
            logger.info(f"    Each frame includes: PlanePositionSequence + PlaneOrientationSequence")
            
            # Add SharedFunctionalGroupsSequence for additional Cornerstone3D compatibility
            # This defines attributes that are common to ALL frames
            shared_item = DicomDataset()
            
            # PlaneOrientationSequence - same for all frames
            if hasattr(datasets[0], 'ImageOrientationPatient'):
                shared_orient_item = DicomDataset()
                shared_orient_item.ImageOrientationPatient = datasets[0].ImageOrientationPatient
                shared_item.PlaneOrientationSequence = Sequence([shared_orient_item])
            
            # PixelMeasuresSequence - pixel spacing and slice thickness
            if hasattr(datasets[0], 'PixelSpacing') or hasattr(datasets[0], 'SliceThickness'):
                pixel_measures_item = DicomDataset()
                if hasattr(datasets[0], 'PixelSpacing'):
                    pixel_measures_item.PixelSpacing = datasets[0].PixelSpacing
                if hasattr(datasets[0], 'SliceThickness'):
                    pixel_measures_item.SliceThickness = datasets[0].SliceThickness
                if hasattr(output_ds, 'SpacingBetweenSlices'):
                    pixel_measures_item.SpacingBetweenSlices = output_ds.SpacingBetweenSlices
                shared_item.PixelMeasuresSequence = Sequence([pixel_measures_item])
            
            output_ds.SharedFunctionalGroupsSequence = Sequence([shared_item])
            logger.info(f"  ✓ Added SharedFunctionalGroupsSequence (common attributes for all frames)")
            logger.info(f"    (Additional defense against Cornerstone3D < v2.0 bugs)")
            
            # Verify frame ordering
            if len(per_frame_seq) > 0:
                first_frame_pos = per_frame_seq[0].PlanePositionSequence[0].ImagePositionPatient if hasattr(per_frame_seq[0], 'PlanePositionSequence') else None
                last_frame_pos = per_frame_seq[-1].PlanePositionSequence[0].ImagePositionPatient if hasattr(per_frame_seq[-1], 'PlanePositionSequence') else None
                
                if first_frame_pos and last_frame_pos:
                    logger.info(f"  ✓ Frame ordering verification:")
                    logger.info(f"    Frame[0] Z = {first_frame_pos[2]} (should match top-level)")
                    logger.info(f"    Frame[{len(per_frame_seq)-1}] Z = {last_frame_pos[2]} (last slice)")
                    
                    # Verify top-level matches Frame[0]
                    if hasattr(output_ds, 'ImagePositionPatient'):
                        if abs(float(output_ds.ImagePositionPatient[2]) - float(first_frame_pos[2])) < 0.001:
                            logger.info(f"    ✅ Top-level ImagePositionPatient matches Frame[0]")
                        else:
                            logger.error(f"    ❌ MISMATCH: Top-level Z={output_ds.ImagePositionPatient[2]} != Frame[0] Z={first_frame_pos[2]}")
            
            logger.info(f"  ✓ Created multi-frame with {total_frame_count} frames (OHIF-compatible)")
            logger.info(f"  ✓ Basic Offset Table included for efficient frame access")
            
            # Create output directory structure
            study_output_dir = os.path.join(output_dir, study_uid)
            os.makedirs(study_output_dir, exist_ok=True)
            
            # Save as single multi-frame file
            output_file = os.path.join(study_output_dir, f"{series_uid}.dcm")
            output_ds.save_as(output_file, write_like_original=False)
            
            logger.info(f"  ✓ Saved multi-frame file: {output_file}")
            processed_series += 1
            total_frames += total_frame_count
            
        except Exception as e:
            logger.error(f"Failed to process series {series_uid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"\nMulti-frame HTJ2K transcoding complete:")
    logger.info(f"  Total series processed: {processed_series}")
    logger.info(f"  Total frames encoded: {total_frames}")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"  Output directory: {output_dir}")
    
    return output_dir
