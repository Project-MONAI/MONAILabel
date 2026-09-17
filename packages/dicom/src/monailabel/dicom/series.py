"""Regular scalar single-frame CT/MR series with explicit LPS ↔ RAS geometry."""

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from numpy.typing import NDArray
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from monailabel.core.errors import DomainError

LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class DicomVolume:
    image: NDArray[np.float32]
    affine: NDArray[np.float64]
    study_uid: str
    series_uid: str
    frame_of_reference_uid: str
    instance_uids: list[str]  # Ordered along the source K axis.
    source_indices: list[int]
    description: str


def read_series(contents: list[bytes]) -> DicomVolume:
    """Reject mixed, irregular, enhanced, and unsupported grids instead of guessing."""
    if not contents or sum(map(len, contents)) > 256 * 1024**2:
        raise DomainError("Choose a nonempty DICOM series below 256 MiB.")
    try:
        slices = [pydicom.dcmread(io.BytesIO(content)) for content in contents]
        source_indices = {id(item): index for index, item in enumerate(slices)}
        first = slices[0]
        shape = (int(first.Columns), int(first.Rows), len(slices))
        if np.prod(shape) > 64 * 1024**2:
            raise DomainError("DICOM series exceeds the 67,108,864 voxel limit.")
        orientation = np.asarray(first.ImageOrientationPatient, dtype=float).reshape(2, 3)
        normal = np.cross(*orientation)
        if not np.allclose(orientation @ orientation.T, np.eye(2), atol=1e-5):
            raise DomainError("DICOM direction cosines must be orthonormal.")
        spacing = np.asarray(first.PixelSpacing, dtype=float)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise DomainError("DICOM pixel spacing must be positive and finite.")
        for item in slices:
            if (
                str(item.Modality) not in {"CT", "MR"}
                or int(item.get("NumberOfFrames", 1)) != 1
                or int(item.SamplesPerPixel) != 1
                or str(item.PhotometricInterpretation) != "MONOCHROME2"
                or item.SeriesInstanceUID != first.SeriesInstanceUID
                or item.StudyInstanceUID != first.StudyInstanceUID
                or item.FrameOfReferenceUID != first.FrameOfReferenceUID
                or (int(item.Columns), int(item.Rows)) != shape[:2]
                or not np.allclose(item.ImageOrientationPatient, orientation.ravel(), atol=1e-5)
                or not np.allclose(item.PixelSpacing, spacing, atol=1e-5)
            ):
                raise DomainError("Import one regular, scalar, single-frame CT or MR series.")
        slices.sort(key=lambda item: float(np.dot(item.ImagePositionPatient, normal)))
        positions = np.asarray([item.ImagePositionPatient for item in slices], dtype=float)
        if not np.isfinite(positions).all():
            raise DomainError("DICOM slice positions must be finite.")
        thickness = float(first.get("SliceThickness", 1))
        delta = normal * thickness if len(slices) == 1 else positions[1] - positions[0]
        if (
            not np.isfinite(delta).all()
            or np.dot(delta, normal) <= 0
            or not np.allclose(delta, normal * np.dot(delta, normal), atol=1e-3)
            or (len(slices) > 1 and not np.allclose(np.diff(positions, axis=0), delta, atol=1e-3))
        ):
            raise DomainError("DICOM slice spacing is irregular or tilted; resample explicitly.")
        uids = [str(item.SOPInstanceUID) for item in slices]
        if len(set(uids)) != len(uids):
            raise DomainError("The series contains duplicate SOP instances.")
        image = np.stack(
            [
                item.pixel_array.astype(np.float32).T * float(item.get("RescaleSlope", 1))
                + float(item.get("RescaleIntercept", 0))
                for item in slices
            ],
            axis=2,
        )
        if not np.isfinite(image).all():
            raise DomainError("DICOM pixels contain nonfinite values.")
        affine = np.eye(4)
        affine[:3, 0] = orientation[0] * spacing[1]
        affine[:3, 1] = orientation[1] * spacing[0]
        affine[:3, 2] = delta
        affine[:3, 3] = positions[0]
        return DicomVolume(
            image,
            LPS_RAS @ affine,
            str(first.StudyInstanceUID),
            str(first.SeriesInstanceUID),
            str(first.FrameOfReferenceUID),
            uids,
            [source_indices[id(item)] for item in slices],
            str(first.get("SeriesDescription", "DICOM series")),
        )
    except DomainError:
        raise
    except (
        ValueError,
        AttributeError,
        KeyError,
        RuntimeError,
        pydicom.errors.InvalidDicomError,
    ) as exc:
        raise DomainError("Could not decode this DICOM series: " + str(exc)) from exc


def derived_ct_series(
    image: NDArray[np.float32],
    affine: NDArray[np.float64],
    name: str,
) -> list[bytes]:
    """Create a derived CT test series from an orthogonal NIfTI grid, preserving HU.

    Patient/study identifiers and acquisition metadata are explicitly synthetic.
    No original clinical metadata can be reconstructed from a NIfTI file.
    """
    if (
        image.ndim != 3
        or not image.size
        or not np.isfinite(image).all()
        or affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
    ):
        raise DomainError("Provide a finite scalar volume and its 4×4 affine.")
    lps = LPS_RAS @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise DomainError("Derived DICOM requires positive voxel spacing.")
    directions = lps[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        raise DomainError("Derived DICOM requires an orthogonal voxel grid.")
    if image.min() < -32768 or image.max() > 32767:
        raise DomainError("The test CT converter supports signed 16-bit HU values.")
    if not np.allclose(image, np.rint(image), atol=1e-4, rtol=0):
        raise DomainError("The test CT converter requires integer HU values; no implicit rounding.")
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    result = []
    for index in range(image.shape[2]):
        uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID, dataset.SOPInstanceUID = CTImageStorage, uid
        dataset.StudyInstanceUID, dataset.SeriesInstanceUID = study, series
        dataset.FrameOfReferenceUID = frame
        dataset.Modality = "CT"
        dataset.ImageType = ["DERIVED", "SECONDARY"]
        dataset.PatientName = "MONAILABEL^TEST"
        dataset.PatientID = "MONAILABEL-TEST-" + name[:40]
        dataset.PatientBirthDate = ""
        dataset.PatientSex = ""
        dataset.PatientIdentityRemoved = "YES"
        dataset.StudyDate, dataset.StudyTime = "20000101", "120000"
        dataset.SeriesDate, dataset.SeriesTime = "20000101", "120000"
        dataset.AcquisitionDate, dataset.AcquisitionTime = "20000101", "120000"
        dataset.StudyID = "MLTEST"
        dataset.AccessionNumber = ""
        dataset.ReferringPhysicianName = ""
        dataset.StudyDescription = "MONAI Label derived research test data"
        dataset.SeriesDescription = (name + " - derived from NIfTI")[:64]
        dataset.DerivationDescription = (
            "Derived from public NIfTI; synthetic patient and acquisition metadata."
        )
        dataset.Manufacturer = "MONAI Label test converter"
        dataset.SeriesNumber, dataset.InstanceNumber = 1, index + 1
        dataset.PositionReferenceIndicator = ""
        dataset.Rows, dataset.Columns = image.shape[1], image.shape[0]
        dataset.ImageOrientationPatient = [float(v) for v in directions[:, :2].T.ravel()]
        dataset.ImagePositionPatient = [float(v) for v in (lps @ [0, 0, index, 1])[:3]]
        dataset.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        dataset.SliceThickness, dataset.SpacingBetweenSlices = float(spacing[2]), float(spacing[2])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 16, 16, 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope, dataset.RescaleIntercept, dataset.RescaleType = 1, 0, "HU"
        dataset.WindowCenter, dataset.WindowWidth = 40, 400
        dataset.PixelData = image[:, :, index].T.astype("<i2").tobytes()
        stream = io.BytesIO()
        dataset.save_as(stream, enforce_file_format=True)
        result.append(stream.getvalue())
    return result
