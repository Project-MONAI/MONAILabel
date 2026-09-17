"""Shared DICOM decoding and source identity for every import entry point."""

import hashlib
import io

import numpy as np
import pydicom

from monailabel.core.models import Asset, DicomSeries, Split
from monailabel.dicom.series import read_series
from monailabel.server.data import nifti_bytes
from monailabel.server.storage import Artifacts


def prepare_series(
    artifacts: Artifacts,
    project_id: str,
    source_url: str,
    contents: list[bytes],
    connection_id: str | None = None,
) -> tuple[Asset, DicomSeries]:
    """Decode and store bytes before the short atomic publication transaction."""
    volume = read_series(contents)
    first = pydicom.dcmread(io.BytesIO(contents[0]), stop_before_pixels=True)
    patient = str(first.get("PatientID", ""))
    issuer = str(first.get("IssuerOfPatientID", ""))
    group = (
        "dicom-patient:" + hashlib.sha256(f"{source_url}\0{issuer}\0{patient}".encode()).hexdigest()
        if patient
        else volume.study_uid
    )
    asset = Asset(
        project_id=project_id,
        name=volume.description + ".nii",
        group_id=group,
        split=Split.POOL,
        kind="volume3d",
        spatial_shape=list(volume.image.shape),
        affine=volume.affine.tolist(),
        source_key=artifacts.put(nifti_bytes(volume.image, volume.affine.tolist())),
        # Match NIfTI decoding order so identical voxels share an artifact identity.
        image_key=artifacts.put_array(np.asfortranarray(volume.image[..., None])),
    )
    series = DicomSeries(
        project_id=project_id,
        asset_id=asset.id,
        study_uid=volume.study_uid,
        series_uid=volume.series_uid,
        frame_of_reference_uid=volume.frame_of_reference_uid,
        instance_uids=volume.instance_uids,
        connection_id=connection_id,
        source_group_id=group,
        source_keys=[artifacts.put(contents[i]) for i in volume.source_indices],
    )
    return asset, series
