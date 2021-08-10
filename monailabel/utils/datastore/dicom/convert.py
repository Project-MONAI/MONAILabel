from typing import Tuple

import pydicom as dicom
from dicom2nifti import settings
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from nibabel.nifti1 import Nifti1Image


class ConverterUtil(object):

    def __init__(self, interp_order: int = 1, fill_value: int = 0) -> None:
        settings.disable_validate_orthogonal()
        settings.enable_resampling()
        settings.set_resample_spline_interpolation_order(interp_order)
        settings.set_resample_padding(fill_value)

    def to_nifti(dicom_dataset: dicom.Dataset, nifti_output_path: str) -> Tuple[Nifti1Image, str]:
        result = dicom_array_to_nifti(dicom_dataset, nifti_output_path)
        return result['NIFTI'], result['NII_FILE']

    def to_dicom(nifti_volume: Nifti1Image) -> dicom.Dataset:
        pass
