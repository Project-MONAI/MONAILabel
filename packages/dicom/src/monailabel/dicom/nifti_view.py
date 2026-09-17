"""Derived DICOM viewing copies; the original NIfTI remains the learning source."""

import io
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from pydicom.valuerep import format_number_as_ds

from monailabel.core.errors import DomainError
from monailabel.dicom.series import LPS_RAS


def viewing_series(
    image: NDArray[np.float32],
    affine: NDArray[np.float64],
    name: str,
    identifier: str,
    progress: Callable[[float], None],
) -> list[bytes]:
    if image.ndim != 3 or not image.size or not np.isfinite(image).all():
        raise DomainError("OHIF requires a finite scalar 3D volume.")
    lps = LPS_RAS @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    if not np.isfinite(lps).all() or np.any(spacing <= 0):
        raise DomainError("The volume needs a valid physical coordinate transform.")
    directions = lps[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        raise DomainError(
            "OHIF requires orthogonal voxel axes. Open this sheared volume in Slicer."
        )
    low, high = float(image.min()), float(image.max())
    integer = low >= -32768 and high <= 32767 and np.equal(image, np.rint(image)).all()
    slope = 1.0 if integer or high == low else (high - low) / 65535
    intercept = 0.0 if integer else low if high == low else low + 32768 * slope
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    files = []
    for index in range(image.shape[2]):
        progress(index / image.shape[2])
        uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SpecificCharacterSet = "ISO_IR 192"
        ds.SOPClassUID, ds.SOPInstanceUID = SecondaryCaptureImageStorage, uid
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = (
            study_uid,
            series_uid,
            frame_uid,
        )
        ds.Modality = "OT"  # NIfTI does not establish a CT/MR acquisition modality.
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.ConversionType = "WSD"
        ds.PatientName, ds.PatientID = "Workspace^Sample", identifier
        ds.PatientBirthDate, ds.PatientSex = "", ""
        ds.StudyDate, ds.StudyTime, ds.AccessionNumber = "", "", ""
        ds.ReferringPhysicianName, ds.StudyID = "", ""
        ds.StudyDescription = "NIfTI viewing copy"
        ds.SeriesDescription = name[:64]
        ds.DerivationDescription = (
            "Workspace NIfTI viewing copy. Geometry preserved; pixels use 16-bit storage "
            "with rescale slope/intercept. Original data is retained for annotation and training."
        )
        ds.Manufacturer = "MONAI Label"
        ds.SeriesNumber, ds.InstanceNumber = 1, index + 1
        ds.Rows, ds.Columns = image.shape[1], image.shape[0]
        ds.ImageOrientationPatient = [float(v) for v in directions[:, :2].T.ravel()]
        ds.ImagePositionPatient = [float(v) for v in (lps @ [0, 0, index, 1])[:3]]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness, ds.SpacingBetweenSlices = float(spacing[2]), float(spacing[2])
        ds.PositionReferenceIndicator = ""
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = (
            format_number_as_ds(slope),
            format_number_as_ds(intercept),
        )
        ds.RescaleType = "US"
        ds.WindowCenter = format_number_as_ds((low + high) / 2)
        ds.WindowWidth = format_number_as_ds(max(high - low, 1.0))
        pixels = np.clip(
            np.rint((image[:, :, index].astype(np.float64) - intercept) / slope), -32768, 32767
        )
        ds.PixelData = pixels.T.astype("<i2").tobytes()
        stream = io.BytesIO()
        ds.save_as(stream, enforce_file_format=True)
        files.append(stream.getvalue())
    return files
