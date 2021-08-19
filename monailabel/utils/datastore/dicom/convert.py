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
import datetime
import logging
import pathlib
from random import randint
from typing import List, Tuple

import numpy as np
import pydicom as dicom
from dicom2nifti import settings
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from nibabel.nifti1 import Nifti1Image
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

logger = logging.getLogger(__name__)


class ConverterUtil(object):
    def __init__(self, interp_order: int = 1, fill_value: int = 0) -> None:
        settings.disable_validate_orthogonal()
        settings.enable_resampling()
        settings.set_resample_spline_interpolation_order(interp_order)
        settings.set_resample_padding(fill_value)

    @staticmethod
    def to_nifti(dicom_dataset: dicom.Dataset, nifti_output_path: str) -> Tuple[Nifti1Image, str]:
        result = dicom_array_to_nifti(dicom_dataset, nifti_output_path)
        return result["NII"], pathlib.Path(result["NII_FILE"]).name

    @staticmethod
    def to_dicom(
        original_data: List[dicom.Dataset],
        seg_img: np.ndarray,
        seg_labels: List[str],
    ) -> dicom.Dataset:

        # Find out the number of DICOM instance datasets
        num_of_dcm_ds = len(original_data)
        logger.info("Number of DICOM instance datasets in the list: {}".format(num_of_dcm_ds))

        # Find out the number of slices in the numpy array
        num_of_img_slices = seg_img.shape[0]
        logger.info("Number of slices in the numpy image: {}".format(num_of_img_slices))

        # Find out the labels
        logger.info("Labels of the segments: {}".format(seg_labels))

        # Find out the unique values in the seg image
        unique_elements = np.unique(seg_img, return_counts=False)

        # if I am not given labels then try to use the unique elements found in the range
        if not seg_labels:
            seg_labels = list(map(str, unique_elements))

        logger.info("Unique values in seg image: {}".format(unique_elements))

        dcmseg_dataset = _create_multiframe_metadata(original_data[0])
        _create_label_segments(dcmseg_dataset, seg_labels)
        _set_pixel_meta(dcmseg_dataset, original_data[0])
        _segslice_from_mhd(dcmseg_dataset, seg_img, original_data, len(seg_labels))

        return dcmseg_dataset


def _safe_get(ds, key):
    """Safely gets the tag value if present from the Dataset and logs failure.

    The safe get method of dict works for str, but not the hex key. The added
    benefit of this funtion is that it logs the failure to get the keyed value.

    Args:
        ds (Dataset): pydicom Dataset
        key (hex | str): Hex code or string name for a key.
    """

    try:
        return ds[key].value
    except KeyError as e:
        logging.error("Failed to get value for key: {}".format(e))
    return ""


def _random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def _create_multiframe_metadata(input_ds) -> Dataset:
    """Creates the DICOM metadata for the multiframe object, e.g. SEG

    Args:
        dicom_file (str or object): The filename or the object type of the file-like the FileDataset was read from.
        input_ds (Dataset): pydicom dataset of original DICOM instance.

    Returns:
        FileDataset: The object with metadata assigned.
    """

    current_date_raw = datetime.datetime.now()
    current_date = current_date_raw.strftime("%Y%m%d")
    current_time = current_date_raw.strftime("%H%M%S.%f")  # long format with micro seconds
    segmentation_series_instance_uid = generate_uid(prefix=None)
    segmentation_sop_instance_uid = generate_uid(prefix=None)

    # Populate required values for file meta information

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    file_meta.MediaStorageSOPInstanceUID = segmentation_sop_instance_uid
    file_meta.ImplementationClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    # create dicom global metadata
    dicom_output = Dataset({}, file_meta=file_meta, preamble=b"\0" * 128)

    # It is important to understand the Types of DICOM attributes when getting from the original
    # dataset, and creating/setting them in the new dataset, .e.g Type 1 is mandatory, though
    # non-conformant instance may not have them, Type 2 present but maybe blank, and Type 3 may
    # be absent.

    # None of Patient module attributes are mandatory.
    # The following are Type 2, present though could be blank
    dicom_output.PatientName = input_ds.get("PatientName", "")  # name is actual suppoted
    dicom_output.add_new(0x00100020, "LO", _safe_get(input_ds, 0x00100020))  # PatientID
    dicom_output.add_new(0x00100030, "DA", _safe_get(input_ds, 0x00100030))  # PatientBirthDate
    dicom_output.add_new(0x00100040, "CS", _safe_get(input_ds, 0x00100040))  # PatientSex
    dicom_output.add_new(0x00104000, "LT", _safe_get(input_ds, "0x00104000"))  # PatientComments

    # For Study module, copy original StudyInstanceUID and other Type 2 study attributes
    # Only Study Instance UID is Type 1, though still may be absent, so try to get
    dicom_output.add_new(0x0020000D, "UI", _safe_get(input_ds, 0x0020000D))  # StudyInstanceUID
    dicom_output.add_new(0x00080020, "DA", input_ds.get("StudyDate", current_date))  # StudyDate
    dicom_output.add_new(0x00080030, "TM", input_ds.get("StudyTime", current_time))  # StudyTime
    dicom_output.add_new(0x00080090, "PN", _safe_get(input_ds, 0x00080090))  # ReferringPhysicianName
    dicom_output.add_new(0x00200010, "SH", _safe_get(input_ds, 0x00200010))  # StudyID
    dicom_output.add_new(0x00080050, "SH", _safe_get(input_ds, 0x00080050))  # AccessionNumber

    # Series module with new attribute values, only Modality and SeriesInstanceUID are Type 1
    dicom_output.add_new(0x00080060, "CS", "SEG")  # Modality
    dicom_output.add_new(0x0020000E, "UI", segmentation_series_instance_uid)  # SeriesInstanceUID
    dicom_output.add_new(0x00200011, "IS", _random_with_n_digits(4))  # SeriesNumber (randomized)
    descr = "MONAI Label generated multiframe DICOMSEG. Not for Clinical use."
    if _safe_get(input_ds, 0x0008103E):
        descr += " for " + _safe_get(input_ds, 0x0008103E)
    dicom_output.add_new(0x0008103E, "LO", descr)  # SeriesDescription
    dicom_output.add_new(0x00080021, "DA", current_date)  # SeriesDate
    dicom_output.add_new(0x00080031, "TM", current_time)  # SeriesTime

    # General Equipment module, only Manufacturer is Type 2, the rest Type 3
    dicom_output.add_new(0x00181000, "LO", "0000")  # DeviceSerialNumber
    dicom_output.add_new(0x00080070, "LO", "MONAI Label")  # Manufacturer
    dicom_output.add_new(0x00081090, "LO", "MONAI Label")  # ManufacturerModelName
    dicom_output.add_new(0x00181020, "LO", "1")  # SoftwareVersions

    # SOP common, only SOPClassUID and SOPInstanceUID are Type 1
    dicom_output.add_new(0x00200013, "IS", 1)  # InstanceNumber
    dicom_output.add_new(0x00080016, "UI", "1.2.840.10008.5.1.4.1.1.66.4")  # SOPClassUID, per DICOM.
    dicom_output.add_new(0x00080018, "UI", segmentation_sop_instance_uid)  # SOPInstanceUID
    dicom_output.add_new(0x00080012, "DA", current_date)  # InstanceCreationDate
    dicom_output.add_new(0x00080013, "TM", current_time)  # InstanceCreationTime

    # General Image module.
    dicom_output.add_new(0x00080008, "CS", ["DERIVED", "PRIMARY"])  # ImageType
    dicom_output.add_new(0x00200020, "CS", "")  # PatientOrientation, forced empty
    # Set content date/time
    dicom_output.ContentDate = current_date
    dicom_output.ContentTime = current_time

    # Image Pixel
    dicom_output.add_new(0x00280002, "US", 1)  # SamplesPerPixel
    dicom_output.add_new(0x00280004, "CS", "MONOCHROME2")  # PhotometricInterpretation

    # Common Instance Reference module
    dicom_output.add_new(0x00081115, "SQ", [Dataset()])  # ReferencedSeriesSequence
    # Set the referenced SeriesInstanceUID
    dicom_output.get(0x00081115)[0].add_new(0x0020000E, "UI", _safe_get(input_ds, 0x0020000E))

    # Multi-frame Dimension Module
    dimension_id = generate_uid(prefix=None)
    dimension_organization_sequence = Sequence()
    dimension_organization_sequence_ds = Dataset()
    dimension_organization_sequence_ds.add_new(0x00209164, "UI", dimension_id)  # DimensionOrganizationUID
    dimension_organization_sequence.append(dimension_organization_sequence_ds)
    dicom_output.add_new(0x00209221, "SQ", dimension_organization_sequence)  # DimensionOrganizationSequence

    dimension_index_sequence = Sequence()
    dimension_index_sequence_ds = Dataset()
    dimension_index_sequence_ds.add_new(0x00209164, "UI", dimension_id)  # DimensionOrganizationUID
    dimension_index_sequence_ds.add_new(0x00209165, "AT", 0x00209153)  # DimensionIndexPointer
    dimension_index_sequence_ds.add_new(0x00209167, "AT", 0x00209153)  # FunctionalGroupPointer
    dimension_index_sequence.append(dimension_index_sequence_ds)
    dicom_output.add_new(0x00209222, "SQ", dimension_index_sequence)  # DimensionIndexSequence

    return dicom_output


def _create_label_segments(dcm_output: Dataset, seg_labels: List[str]) -> None:
    """ "Creates the segments with the given labels"""

    def create_label_segment(label, name):
        """Creates segment labels"""
        segment = Dataset()
        segment.add_new(0x00620004, "US", int(label))  # SegmentNumber
        segment.add_new(0x00620005, "LO", name)  # SegmentLabel
        segment.add_new(0x00620009, "LO", "AI Organ Segmentation")  # SegmentAlgorithmName
        segment.SegmentAlgorithmType = "AUTOMATIC"  # SegmentAlgorithmType
        segment.add_new(0x0062000D, "US", [128, 174, 128])  # RecommendedDisplayCIELabValue
        # create SegmentedPropertyCategoryCodeSequence
        segmented_property_category_code_sequence = Sequence()
        segmented_property_category_code_sequence_ds = Dataset()
        segmented_property_category_code_sequence_ds.add_new(0x00080100, "SH", "T-D0050")  # CodeValue
        segmented_property_category_code_sequence_ds.add_new(0x00080102, "SH", "SRT")  # CodingSchemeDesignator
        segmented_property_category_code_sequence_ds.add_new(0x00080104, "LO", "Anatomical Structure")  # CodeMeaning
        segmented_property_category_code_sequence.append(segmented_property_category_code_sequence_ds)
        segment.SegmentedPropertyCategoryCodeSequence = segmented_property_category_code_sequence
        # create SegmentedPropertyTypeCodeSequence
        segmented_property_type_code_sequence = Sequence()
        segmented_property_type_code_sequence_ds = Dataset()
        segmented_property_type_code_sequence_ds.add_new(0x00080100, "SH", "T-D0050")  # CodeValue
        segmented_property_type_code_sequence_ds.add_new(0x00080102, "SH", "SRT")  # CodingSchemeDesignator
        segmented_property_type_code_sequence_ds.add_new(0x00080104, "LO", "Organ")  # CodeMeaning
        segmented_property_type_code_sequence.append(segmented_property_type_code_sequence_ds)
        segment.SegmentedPropertyTypeCodeSequence = segmented_property_type_code_sequence
        return segment

    segments = Sequence()
    # Assumes the label starts at 1 and increment sequentially.
    # TODO: This part needs to be more deteministic, e.g. with a dict.
    for lb, name in enumerate(seg_labels, 1):
        segment = create_label_segment(lb, name)
        segments.append(segment)
    dcm_output.add_new(0x00620002, "SQ", segments)  # SegmentSequence


def _create_frame_meta(input_ds, label, ref_instances, dim_idx_val, instance_num):
    """Creates the metadata for the each frame"""

    sop_inst_uid = _safe_get(input_ds, 0x00080018)  # SOPInstanceUID
    source_instance_sop_class = _safe_get(input_ds, 0x00080016)  # SOPClassUID

    # add frame to Referenced Image Sequence
    frame_ds = Dataset()
    reference_instance = Dataset()
    reference_instance.add_new(0x00081150, "UI", source_instance_sop_class)  # ReferencedSOPClassUID
    reference_instance.add_new(0x00081155, "UI", sop_inst_uid)  # ReferencedSOPInstanceUID

    ref_instances.append(reference_instance)
    ############################
    # CREATE METADATA
    ############################
    # create DerivationImageSequence within Per-frame Functional Groups sequence
    derivation_image_sequence = Sequence()
    derivation_image = Dataset()
    # create SourceImageSequence within DerivationImageSequence
    source_image_sequence = Sequence()
    source_image = Dataset()
    # TODO if CT multi-frame
    # sourceImage.add_new(0x00081160, 'IS', inputFrameCounter + 1) # Referenced Frame Number
    source_image.add_new(0x00081150, "UI", source_instance_sop_class)  # ReferencedSOPClassUID
    source_image.add_new(0x00081155, "UI", sop_inst_uid)  # ReferencedSOPInstanceUID
    # create PurposeOfReferenceCodeSequence within SourceImageSequence
    purpose_of_reference_code_sequence = Sequence()
    purpose_of_reference_code = Dataset()
    purpose_of_reference_code.add_new(0x00080100, "SH", "121322")  # CodeValue
    purpose_of_reference_code.add_new(0x00080102, "SH", "DCM")  # CodingSchemeDesignator
    purpose_of_reference_code.add_new(0x00080104, "LO", "Anatomical Stucture")  # CodeMeaning
    purpose_of_reference_code_sequence.append(purpose_of_reference_code)
    source_image.add_new(0x0040A170, "SQ", purpose_of_reference_code_sequence)  # PurposeOfReferenceCodeSequence
    source_image_sequence.append(source_image)  # AEH Beck commentout
    # create DerivationCodeSequence within DerivationImageSequence
    derivation_code_sequence = Sequence()
    derivation_code = Dataset()
    derivation_code.add_new(0x00080100, "SH", "113076")  # CodeValue
    derivation_code.add_new(0x00080102, "SH", "DCM")  # CodingSchemeDesignator
    derivation_code.add_new(0x00080104, "LO", "Segmentation")  # CodeMeaning
    derivation_code_sequence.append(derivation_code)
    derivation_image.add_new(0x00089215, "SQ", derivation_code_sequence)  # DerivationCodeSequence
    derivation_image.add_new(0x00082112, "SQ", source_image_sequence)  # SourceImageSequence
    derivation_image_sequence.append(derivation_image)
    frame_ds.add_new(0x00089124, "SQ", derivation_image_sequence)  # DerivationImageSequence
    # create FrameContentSequence within Per-frame Functional Groups sequence
    frame_content = Sequence()
    dimension_index_values = Dataset()
    dimension_index_values.add_new(0x00209157, "UL", [dim_idx_val, instance_num])  # DimensionIndexValues
    frame_content.append(dimension_index_values)
    frame_ds.add_new(0x00209111, "SQ", frame_content)  # FrameContentSequence
    # create PlanePositionSequence within Per-frame Functional Groups sequence
    plane_position_sequence = Sequence()
    image_position_patient = Dataset()
    image_position_patient.add_new(0x00200032, "DS", _safe_get(input_ds, 0x00200032))  # ImagePositionPatient
    plane_position_sequence.append(image_position_patient)
    frame_ds.add_new(0x00209113, "SQ", plane_position_sequence)  # PlanePositionSequence
    # create PlaneOrientationSequence within Per-frame Functional Groups sequence
    plane_orientation_sequence = Sequence()
    image_orientation_patient = Dataset()
    image_orientation_patient.add_new(0x00200037, "DS", _safe_get(input_ds, 0x00200037))  # ImageOrientationPatient
    plane_orientation_sequence.append(image_orientation_patient)
    frame_ds.add_new(0x00209116, "SQ", plane_orientation_sequence)  # PlaneOrientationSequence
    # create SegmentIdentificationSequence within Per-frame Functional Groups sequence
    segment_identification_sequence = Sequence()
    referenced_segment_number = Dataset()
    # TODO lop over label and only get pixel with that value
    referenced_segment_number.add_new(0x0062000B, "US", label)  # ReferencedSegmentNumber, which label is this frame
    segment_identification_sequence.append(referenced_segment_number)
    frame_ds.add_new(0x0062000A, "SQ", segment_identification_sequence)  # SegmentIdentificationSequence
    return frame_ds


def _set_pixel_meta(dcmseg_output: Dataset, input_ds: Dataset) -> None:
    """Sets the pixel metadata in the DICOM object"""

    dcmseg_output.Rows = input_ds.Rows
    dcmseg_output.Columns = input_ds.Columns
    dcmseg_output.BitsAllocated = 1  # add_new(0x00280100, 'US', 8) # Bits allocated
    dcmseg_output.BitsStored = 1
    dcmseg_output.HighBit = 0
    dcmseg_output.PixelRepresentation = 0
    # dicomOutput.PixelRepresentation = input_ds.PixelRepresentation
    dcmseg_output.SamplesPerPixel = 1
    dcmseg_output.ImageType = "DERIVED\\PRIMARY"
    dcmseg_output.ContentLabel = "SEGMENTATION"
    dcmseg_output.ContentDescription = ""
    dcmseg_output.ContentCreatorName = ""
    dcmseg_output.LossyImageCompression = "00"
    dcmseg_output.SegmentationType = "BINARY"
    dcmseg_output.MaximumFractionalValue = 1
    dcmseg_output.SharedFunctionalGroupsSequence = Sequence()
    dcmseg_output.PixelPaddingValue = 0
    # Try to get the attributes from the original.
    # Even though they are Type 1 and 2, can still be absent
    dcmseg_output.PixelSpacing = copy.deepcopy(input_ds.get("PixelSpacing", None))
    dcmseg_output.SliceThickness = input_ds.get("SliceThickness", "")
    dcmseg_output.RescaleSlope = 1
    dcmseg_output.RescaleIntercept = 0
    # Set the transfer syntax
    dcmseg_output.is_little_endian = False  # True
    dcmseg_output.is_implicit_VR = False  # True


def _segslice_from_mhd(dcm_output: Dataset, seg_img: np.ndarray, input_ds: Dataset, num_labels: int) -> None:
    """Sets the pixel data from the input numpy image"""

    # add frames
    out_frame_counter = 0
    out_frames = Sequence()

    out_pixels: List[np.ndarray] = []

    reference_instances = Sequence()

    for img_slice in range(seg_img.shape[0]):

        dim_idx_val = 0

        for label in range(1, num_labels + 1):

            dim_idx_val += 1

            frame_meta = _create_frame_meta(input_ds[img_slice], label, reference_instances, dim_idx_val, img_slice)

            out_frames.append(frame_meta)
            logging.info(
                "img slice {}, label {}, frame {}, img pos {}".format(
                    img_slice, label, out_frame_counter, _safe_get(input_ds[img_slice], 0x00200032)
                )
            )
            seg_slice = np.zeros((1, seg_img.shape[1], seg_img.shape[2]), dtype=bool)

            seg_slice[np.expand_dims(seg_img[img_slice, ...] == label, 0)] = 1

            out_pixels.append(seg_slice)

            out_frame_counter = out_frame_counter + 1

    out_pixels_arr: np.ndarray = np.concatenate(out_pixels, axis=0)

    dcm_output.add_new(0x52009230, "SQ", out_frames)  # PerFrameFunctionalGroupsSequence
    dcm_output.NumberOfFrames = out_frame_counter
    dcm_output.PixelData = np.packbits(np.flip(np.reshape(out_pixels_arr.astype(bool), (-1, 8)), 1)).tostring()

    dcm_output.get(0x00081115)[0].add_new(0x0008114A, "SQ", reference_instances)  # ReferencedInstanceSequence

    # create shared  Functional Groups sequence
    shared_functional_groups = Sequence()
    shared_functional_groups_ds = Dataset()

    plane_orientation_seq = Sequence()
    plane_orientation_ds = Dataset()
    plane_orientation_ds.add_new("0x00200037", "DS", _safe_get(input_ds[0], 0x00200037))  # ImageOrientationPatient
    plane_orientation_seq.append(plane_orientation_ds)
    shared_functional_groups_ds.add_new("0x00209116", "SQ", plane_orientation_seq)  # PlaneOrientationSequence

    pixel_measures_sequence = Sequence()
    pixel_measures_ds = Dataset()
    pixel_measures_ds.add_new("0x00280030", "DS", _safe_get(input_ds[0], "0x00280030"))  # PixelSpacing
    if input_ds[0].get("SpacingBetweenSlices", ""):
        pixel_measures_ds.add_new("0x00180088", "DS", input_ds[0].get("SpacingBetweenSlices", ""))
    pixel_measures_ds.add_new("0x00180050", "DS", _safe_get(input_ds[0], "0x00180050"))  # SliceThickness
    pixel_measures_sequence.append(pixel_measures_ds)
    shared_functional_groups_ds.add_new("0x00289110", "SQ", pixel_measures_sequence)  # PixelMeasuresSequence

    shared_functional_groups.append(shared_functional_groups_ds)

    dcm_output.add_new(0x52009229, "SQ", shared_functional_groups)  # SharedFunctionalGroupsSequence
