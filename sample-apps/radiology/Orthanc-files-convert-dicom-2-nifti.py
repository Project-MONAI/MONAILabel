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
import glob
import json
import logging
import os
import pathlib
import shutil
import tempfile
import time

import numpy as np
import pydicom_seg
import SimpleITK
from monai.data import write_nifti
from monai.transforms import LoadImage
from pydicom.filereader import dcmread

from monailabel.datastore.utils.colors import GENERIC_ANATOMY_COLORS
from monailabel.transform.writer import write_itk
from monailabel.utils.others.generic import run_command

logger = logging.getLogger(__name__)


def dicom_to_nifti(series_dir, is_seg=False):
    start = time.time()

    if is_seg:
        output_file = itk_dicom_seg_to_image(series_dir)
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


def binary_to_image(reference_image, label, dtype=np.uint16, file_ext=".nii.gz", use_itk=True):
    start = time.time()

    image_np, meta_dict = LoadImage()(reference_image)
    label_np = np.fromfile(label, dtype=dtype)

    logger.info(f"Image: {image_np.shape}")
    logger.info(f"Label: {label_np.shape}")

    label_np = label_np.reshape(image_np.shape, order="F")
    logger.info(f"Label (reshape): {label_np.shape}")

    output_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
    affine = meta_dict.get("affine")
    if use_itk:
        write_itk(label_np, output_file, affine=affine, dtype=None, compress=True)
    else:
        write_nifti(label_np, output_file, affine=affine)

    logger.info(f"binary_to_image latency : {time.time() - start} (sec)")
    return output_file


def itk_image_to_dicom_seg(label, series_dir, template):
    output_file = tempfile.NamedTemporaryFile(suffix=".dcm").name
    meta_data = tempfile.NamedTemporaryFile(suffix=".json").name
    with open(meta_data, "w") as fp:
        json.dump(template, fp)

    command = "itkimage2segimage"
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


def itk_dicom_seg_to_image(label, output_type="nifti"):
    # TODO:: Currently supports only one file
    pre = label.split("/")[-2]
    filename = label if not os.path.isdir(label) else os.path.join(label, os.listdir(label)[0])
    command = "./segimage2itkimage"
    args = [
        "--inputDICOM",
        filename,
        "--outputType",
        output_type,
        "--prefix",
        "segment",
        "--outputDirectory",
        label,
    ]
    run_command(command, args)
    output_files = [f for f in os.listdir(label) if f.startswith("segment") and f.endswith(".nii.gz")]
    if not output_files:
        logger.warning(f"Failed to convert DICOM-SEG {label} to NIFTI")
        return None


def main():
    # First, download the images from Orthanc
    # Second: Download the segimage2itkimage - This can be downloaded starting monailabel server with orthanc
    # CONVERT DICOM INTO NIFTI - IMAGES:
    # plastimatch convert --input LIDCIDRI0119/CT/ --output-img LIDCIDRI0119/LIDCIDRI0119.nii.gz
    # Then convert labels in nifti
    image_names = glob.glob("/home/andres/Downloads/dicom-test/final/*")
    for l in image_names:
        print(f'Converting image: {l.split("/")[-1]}')
        dicom_to_nifti(l + "/SEG/", is_seg=True)


if __name__ == "__main__":
    main()
