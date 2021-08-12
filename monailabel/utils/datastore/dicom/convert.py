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
            original_data: dicom.Dataset,
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
    return ''


def _random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10**(n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def _create_multiframe_metadata(input_ds) -> Dataset:
    """Creates the DICOM metadata for the multiframe object, e.g. SEG

    Args:
        dicom_file (str or object): The filename or the object type of the file-like the FileDataset was read from.
        input_ds (Dataset): pydicom dataset of original DICOM instance.

    Returns:
        FileDataset: The object with metadata assigned.
    """

    currentDateRaw = datetime.datetime.now()
    currentDate = currentDateRaw.strftime('%Y%m%d')
    currentTime = currentDateRaw.strftime('%H%M%S.%f')  # long format with micro seconds
    segmentationSeriesInstanceUID = generate_uid(prefix=None)
    segmentationSOPInstanceUID = generate_uid(prefix=None)

    # Populate required values for file meta information

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
    file_meta.MediaStorageSOPInstanceUID = segmentationSOPInstanceUID
    file_meta.ImplementationClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    # create dicom global metadata
    dicomOutput = Dataset({}, file_meta=file_meta, preamble=b"\0" * 128)

    # It is important to understand the Types of DICOM attributes when getting from the original
    # dataset, and creating/setting them in the new dataset, .e.g Type 1 is mandatory, though
    # non-conformant instance may not have them, Type 2 present but maybe blank, and Type 3 may
    # be absent.

    # None of Patient module attributes are mandatory.
    # The following are Type 2, present though could be blank
    dicomOutput.PatientName = input_ds.get('PatientName', '')  # name is actual suppoted
    dicomOutput.add_new(0x00100020, 'LO', _safe_get(input_ds, 0x00100020))  # PatientID
    dicomOutput.add_new(0x00100030, 'DA', _safe_get(input_ds, 0x00100030))  # PatientBirthDate
    dicomOutput.add_new(0x00100040, 'CS', _safe_get(input_ds, 0x00100040))  # PatientSex
    dicomOutput.add_new(0x00104000, 'LT', _safe_get(input_ds, '0x00104000'))  # PatientComments

    # For Study module, copy original StudyInstanceUID and other Type 2 study attributes
    # Only Study Instance UID is Type 1, though still may be absent, so try to get
    dicomOutput.add_new(0x0020000D, 'UI', _safe_get(input_ds, 0x0020000D))  # StudyInstanceUID
    dicomOutput.add_new(0x00080020, 'DA', input_ds.get('StudyDate', currentDate))  # StudyDate
    dicomOutput.add_new(0x00080030, 'TM', input_ds.get('StudyTime', currentTime))  # StudyTime
    dicomOutput.add_new(0x00080090, 'PN', _safe_get(input_ds, 0x00080090))  # ReferringPhysicianName
    dicomOutput.add_new(0x00200010, 'SH', _safe_get(input_ds, 0x00200010))  # StudyID
    dicomOutput.add_new(0x00080050, 'SH', _safe_get(input_ds, 0x00080050))  # AccessionNumber

    # Series module with new attribute values, only Modality and SeriesInstanceUID are Type 1
    dicomOutput.add_new(0x00080060, 'CS', 'SEG')  # Modality
    dicomOutput.add_new(0x0020000E, 'UI', segmentationSeriesInstanceUID)  # SeriesInstanceUID
    dicomOutput.add_new(0x00200011, 'IS', _random_with_n_digits(4))  # SeriesNumber (randomized)
    descr = "MONAI Label generated multiframe DICOMSEG. Not for Clinical use."
    if _safe_get(input_ds, 0x0008103e):
        descr += " for " + _safe_get(input_ds, 0x0008103e)
    dicomOutput.add_new(0x0008103e, 'LO', descr)  # SeriesDescription
    dicomOutput.add_new(0x00080021, 'DA', currentDate)  # SeriesDate
    dicomOutput.add_new(0x00080031, 'TM', currentTime)  # SeriesTime

    # General Equipment module, only Manufacturer is Type 2, the rest Type 3
    dicomOutput.add_new(0x00181000, 'LO', '0000')  # DeviceSerialNumber
    dicomOutput.add_new(0x00080070, 'LO', 'NVIDIA')  # Manufacturer
    dicomOutput.add_new(0x00081090, 'LO', 'CLARA')  # ManufacturerModelName
    dicomOutput.add_new(0x00181020, 'LO', '1')  # SoftwareVersions

    # SOP common, only SOPClassUID and SOPInstanceUID are Type 1
    dicomOutput.add_new(0x00200013, 'IS', 1)  # InstanceNumber
    dicomOutput.add_new(0x00080016, 'UI', '1.2.840.10008.5.1.4.1.1.66.4')  # SOPClassUID, per DICOM.
    dicomOutput.add_new(0x00080018, 'UI', segmentationSOPInstanceUID)  # SOPInstanceUID
    dicomOutput.add_new(0x00080012, 'DA', currentDate)  # InstanceCreationDate
    dicomOutput.add_new(0x00080013, 'TM', currentTime)  # InstanceCreationTime

    # General Image module.
    dicomOutput.add_new(0x00080008, 'CS', ['DERIVED', 'PRIMARY'])  # ImageType
    dicomOutput.add_new(0x00200020, 'CS', '')  # PatientOrientation, forced empty
    # Set content date/time
    dicomOutput.ContentDate = currentDate
    dicomOutput.ContentTime = currentTime

    # Image Pixel
    dicomOutput.add_new(0x00280002, 'US', 1)  # SamplesPerPixel
    dicomOutput.add_new(0x00280004, 'CS', 'MONOCHROME2')  # PhotometricInterpretation

    # Common Instance Reference module
    dicomOutput.add_new(0x00081115, 'SQ', [Dataset()])  # ReferencedSeriesSequence
    # Set the referenced SeriesInstanceUID
    dicomOutput.get(0x00081115)[0].add_new(0x0020000E, 'UI', _safe_get(input_ds, 0x0020000E))

    # Multi-frame Dimension Module
    dimensionID = generate_uid(prefix=None)
    dimensionOragnizationSequence = Sequence()
    dimensionOragnizationSequenceDS = Dataset()
    dimensionOragnizationSequenceDS.add_new(0x00209164, 'UI', dimensionID)  # DimensionOrganizationUID
    dimensionOragnizationSequence.append(dimensionOragnizationSequenceDS)
    dicomOutput.add_new(0x00209221, 'SQ', dimensionOragnizationSequence)  # DimensionOrganizationSequence

    dimensionIndexSequence = Sequence()
    dimensionIndexSequenceDS = Dataset()
    dimensionIndexSequenceDS.add_new(0x00209164, 'UI', dimensionID)  # DimensionOrganizationUID
    dimensionIndexSequenceDS.add_new(0x00209165, 'AT', 0x00209153)  # DimensionIndexPointer
    dimensionIndexSequenceDS.add_new(0x00209167, 'AT', 0x00209153)  # FunctionalGroupPointer
    dimensionIndexSequence.append(dimensionIndexSequenceDS)
    dicomOutput.add_new(0x00209222, 'SQ', dimensionIndexSequence)  # DimensionIndexSequence

    return dicomOutput


def _create_label_segments(dcm_output: Dataset, seg_labels: List[str]) -> None:
    """"Creates the segments with the given labels
    """

    def create_label_segment(label, name):
        """Creates segment labels
        """
        segment = Dataset()
        segment.add_new(0x00620004, 'US', int(label))  # SegmentNumber
        segment.add_new(0x00620005, 'LO', name)  # SegmentLabel
        segment.add_new(0x00620009, 'LO', 'AI Organ Segmentation')  # SegmentAlgorithmName
        segment.SegmentAlgorithmType = 'AUTOMATIC'  # SegmentAlgorithmType
        segment.add_new(0x0062000d, 'US', [128, 174, 128])  # RecommendedDisplayCIELabValue
        # create SegmentedPropertyCategoryCodeSequence
        segmentedPropertyCategoryCodeSequence = Sequence()
        segmentedPropertyCategoryCodeSequenceDS = Dataset()
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080100, 'SH', 'T-D0050')  # CodeValue
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080102, 'SH', 'SRT')  # CodingSchemeDesignator
        segmentedPropertyCategoryCodeSequenceDS.add_new(0x00080104, 'LO', 'Anatomical Structure')  # CodeMeaning
        segmentedPropertyCategoryCodeSequence.append(segmentedPropertyCategoryCodeSequenceDS)
        segment.SegmentedPropertyCategoryCodeSequence = segmentedPropertyCategoryCodeSequence
        # create SegmentedPropertyTypeCodeSequence
        segmentedPropertyTypeCodeSequence = Sequence()
        segmentedPropertyTypeCodeSequenceDS = Dataset()
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080100, 'SH', 'T-D0050')  # CodeValue
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080102, 'SH', 'SRT')  # CodingSchemeDesignator
        segmentedPropertyTypeCodeSequenceDS.add_new(0x00080104, 'LO', 'Organ')  # CodeMeaning
        segmentedPropertyTypeCodeSequence.append(segmentedPropertyTypeCodeSequenceDS)
        segment.SegmentedPropertyTypeCodeSequence = segmentedPropertyTypeCodeSequence
        return segment

    segments = Sequence()
    # Assumes the label starts at 1 and increment sequentially.
    # TODO: This part needs to be more deteministic, e.g. with a dict.
    for lb, name in enumerate(seg_labels, 1):
        segment = create_label_segment(lb, name)
        segments.append(segment)
    dcm_output.add_new(0x00620002, 'SQ', segments)  # SegmentSequence


def _create_frame_meta(input_ds, label, ref_instances, dimIdxVal, instance_num):
    """Creates the metadata for the each frame
    """

    sop_inst_uid = _safe_get(input_ds, 0x00080018)  # SOPInstanceUID
    sourceInstanceSOPClass = _safe_get(input_ds, 0x00080016)  # SOPClassUID

    # add frame to Referenced Image Sequence
    frame_ds = Dataset()
    referenceInstance = Dataset()
    referenceInstance.add_new(0x00081150, 'UI', sourceInstanceSOPClass)  # ReferencedSOPClassUID
    referenceInstance.add_new(0x00081155, 'UI', sop_inst_uid)  # ReferencedSOPInstanceUID

    ref_instances.append(referenceInstance)
    ############################
    # CREATE METADATA
    ############################
    # create DerivationImageSequence within Per-frame Functional Groups sequence
    derivationImageSequence = Sequence()
    derivationImage = Dataset()
    # create SourceImageSequence within DerivationImageSequence
    sourceImageSequence = Sequence()
    sourceImage = Dataset()
    # TODO if CT multi-frame
    # sourceImage.add_new(0x00081160, 'IS', inputFrameCounter + 1) # Referenced Frame Number
    sourceImage.add_new(0x00081150, 'UI', sourceInstanceSOPClass)  # ReferencedSOPClassUID
    sourceImage.add_new(0x00081155, 'UI', sop_inst_uid)  # ReferencedSOPInstanceUID
    # create PurposeOfReferenceCodeSequence within SourceImageSequence
    purposeOfReferenceCodeSequence = Sequence()
    purposeOfReferenceCode = Dataset()
    purposeOfReferenceCode.add_new(0x00080100, 'SH', '121322')  # CodeValue
    purposeOfReferenceCode.add_new(0x00080102, 'SH', 'DCM')  # CodingSchemeDesignator
    purposeOfReferenceCode.add_new(0x00080104, 'LO', 'Anatomical Stucture')  # CodeMeaning
    purposeOfReferenceCodeSequence.append(purposeOfReferenceCode)
    sourceImage.add_new(0x0040a170, 'SQ', purposeOfReferenceCodeSequence)  # PurposeOfReferenceCodeSequence
    sourceImageSequence.append(sourceImage)  # AEH Beck commentout
    # create DerivationCodeSequence within DerivationImageSequence
    derivationCodeSequence = Sequence()
    derivationCode = Dataset()
    derivationCode.add_new(0x00080100, 'SH', '113076')  # CodeValue
    derivationCode.add_new(0x00080102, 'SH', 'DCM')  # CodingSchemeDesignator
    derivationCode.add_new(0x00080104, 'LO', 'Segmentation')  # CodeMeaning
    derivationCodeSequence.append(derivationCode)
    derivationImage.add_new(0x00089215, 'SQ', derivationCodeSequence)  # DerivationCodeSequence
    derivationImage.add_new(0x00082112, 'SQ', sourceImageSequence)  # SourceImageSequence
    derivationImageSequence.append(derivationImage)
    frame_ds.add_new(0x00089124, 'SQ', derivationImageSequence)  # DerivationImageSequence
    # create FrameContentSequence within Per-frame Functional Groups sequence
    frameContent = Sequence()
    dimensionIndexValues = Dataset()
    dimensionIndexValues.add_new(0x00209157, 'UL', [dimIdxVal, instance_num])  # DimensionIndexValues
    frameContent.append(dimensionIndexValues)
    frame_ds.add_new(0x00209111, 'SQ', frameContent)  # FrameContentSequence
    # create PlanePositionSequence within Per-frame Functional Groups sequence
    planePositionSequence = Sequence()
    imagePositionPatient = Dataset()
    imagePositionPatient.add_new(0x00200032, 'DS', _safe_get(input_ds, 0x00200032))  # ImagePositionPatient
    planePositionSequence.append(imagePositionPatient)
    frame_ds.add_new(0x00209113, 'SQ', planePositionSequence)  # PlanePositionSequence
    # create PlaneOrientationSequence within Per-frame Functional Groups sequence
    planeOrientationSequence = Sequence()
    imageOrientationPatient = Dataset()
    imageOrientationPatient.add_new(0x00200037, 'DS', _safe_get(input_ds, 0x00200037))  # ImageOrientationPatient
    planeOrientationSequence.append(imageOrientationPatient)
    frame_ds.add_new(0x00209116, 'SQ', planeOrientationSequence)  # PlaneOrientationSequence
    # create SegmentIdentificationSequence within Per-frame Functional Groups sequence
    segmentIdentificationSequence = Sequence()
    referencedSegmentNumber = Dataset()
    # TODO lop over label and only get pixel with that value
    referencedSegmentNumber.add_new(0x0062000B, 'US', label)  # ReferencedSegmentNumber, which label is this frame
    segmentIdentificationSequence.append(referencedSegmentNumber)
    frame_ds.add_new(0x0062000A, 'SQ', segmentIdentificationSequence)  # SegmentIdentificationSequence
    return frame_ds


def _set_pixel_meta(dcmseg_output: Dataset, input_ds: Dataset) -> None:
    """Sets the pixel metadata in the DICOM object
    """

    dcmseg_output.Rows = input_ds.Rows
    dcmseg_output.Columns = input_ds.Columns
    dcmseg_output.BitsAllocated = 1  # add_new(0x00280100, 'US', 8) # Bits allocated
    dcmseg_output.BitsStored = 1
    dcmseg_output.HighBit = 0
    dcmseg_output.PixelRepresentation = 0
    # dicomOutput.PixelRepresentation = input_ds.PixelRepresentation
    dcmseg_output.SamplesPerPixel = 1
    dcmseg_output.ImageType = 'DERIVED\PRIMARY'
    dcmseg_output.ContentLabel = 'SEGMENTATION'
    dcmseg_output.ContentDescription = ''
    dcmseg_output.ContentCreatorName = ''
    dcmseg_output.LossyImageCompression = '00'
    dcmseg_output.SegmentationType = 'BINARY'
    dcmseg_output.MaximumFractionalValue = 1
    dcmseg_output.SharedFunctionalGroupsSequence = Sequence()
    dcmseg_output.PixelPaddingValue = 0
    # Try to get the attributes from the original.
    # Even though they are Type 1 and 2, can still be absent
    dcmseg_output.PixelSpacing = copy.deepcopy(input_ds.get('PixelSpacing', None))
    dcmseg_output.SliceThickness = input_ds.get('SliceThickness', '')
    dcmseg_output.RescaleSlope = 1
    dcmseg_output.RescaleIntercept = 0
    # Set the transfer syntax
    dcmseg_output.is_little_endian = False  # True
    dcmseg_output.is_implicit_VR = False  # True


def _segslice_from_mhd(dcm_output: Dataset, seg_img: np.ndarray, input_ds: Dataset, num_labels: int) -> None:
    """Sets the pixel data from the input numpy image
    """

    # add frames
    out_frame_counter = 0
    out_frames = Sequence()

    out_pixels = None

    referenceInstances = Sequence()

    for img_slice in range(seg_img.shape[0]):

        dimIdxVal = 0

        for label in range(1, num_labels + 1):

            dimIdxVal += 1

            frame_meta = _create_frame_meta(input_ds[img_slice], label, referenceInstances, dimIdxVal, img_slice)

            out_frames.append(frame_meta)
            logging.info("img slice {}, label {}, frame {}, img pos {}".format(
                img_slice, label, out_frame_counter, _safe_get(input_ds[img_slice], 0x00200032)))
            seg_slice = np.zeros((1, seg_img.shape[1], seg_img.shape[2]), dtype=np.bool)

            seg_slice[np.expand_dims(seg_img[img_slice, ...] == label, 0)] = 1

            if out_pixels is None:
                out_pixels = seg_slice
            else:
                out_pixels = np.concatenate((out_pixels, seg_slice), axis=0)

            out_frame_counter = out_frame_counter + 1

    dcm_output.add_new(0x52009230, 'SQ', out_frames)  # PerFrameFunctionalGroupsSequence
    dcm_output.NumberOfFrames = out_frame_counter
    dcm_output.PixelData = np.packbits(np.flip(np.reshape(out_pixels.astype(np.bool), (-1, 8)), 1)).tostring()

    dcm_output.get(0x00081115)[0].add_new(0x0008114A, 'SQ', referenceInstances)  # ReferencedInstanceSequence

    # create shared  Functional Groups sequence
    sharedFunctionalGroups = Sequence()
    sharedFunctionalGroupsDS = Dataset()

    planeOrientationSeq = Sequence()
    planeOrientationDS = Dataset()
    planeOrientationDS.add_new('0x00200037', 'DS', _safe_get(input_ds[0], 0x00200037))  # ImageOrientationPatient
    planeOrientationSeq.append(planeOrientationDS)
    sharedFunctionalGroupsDS.add_new('0x00209116', 'SQ', planeOrientationSeq)  # PlaneOrientationSequence

    pixelMeasuresSequence = Sequence()
    pixelMeasuresDS = Dataset()
    pixelMeasuresDS.add_new('0x00280030', 'DS', _safe_get(input_ds[0], '0x00280030'))  # PixelSpacing
    if input_ds[0].get('SpacingBetweenSlices', ''):
        pixelMeasuresDS.add_new('0x00180088', 'DS', input_ds[0].get('SpacingBetweenSlices', ''))  # SpacingBetweenSlices
    pixelMeasuresDS.add_new('0x00180050', 'DS', _safe_get(input_ds[0], '0x00180050'))  # SliceThickness
    pixelMeasuresSequence.append(pixelMeasuresDS)
    sharedFunctionalGroupsDS.add_new('0x00289110', 'SQ', pixelMeasuresSequence)  # PixelMeasuresSequence

    sharedFunctionalGroups.append(sharedFunctionalGroupsDS)

    dcm_output.add_new(0x52009229, 'SQ', sharedFunctionalGroups)  # SharedFunctionalGroupsSequence
