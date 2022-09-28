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

import json
import logging
import os
import pathlib
import tempfile
import time

import numpy as np
import pydicom
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


def nifti_to_dicom_seg(series_dir, label, label_info, file_ext="*", use_itk=True):
    start = time.time()

    label_np, meta_dict = LoadImage()(label)
    unique_labels = np.unique(label_np.flatten()).astype(np.int)
    unique_labels = unique_labels[unique_labels != 0]

    segment_attributes = []
    for i, idx in enumerate(unique_labels):
        info = label_info[i] if label_info and i < len(label_info) else {}
        name = info.get("name", "unknown")
        description = info.get("description", "Unknown")
        rgb = list(info.get("color", GENERIC_ANATOMY_COLORS.get(name, (255, 0, 0))))[0:3]
        rgb = [int(x) for x in rgb]

        logger.info(f"{i} => {idx} => {name}")

        segment_attribute = info.get(
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
        segment_attributes.append(segment_attribute)

    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": "Segmentation",
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "MONAI Label - Image segmentation",
        "ClinicalTrialCoordinatingCenterName": "MONAI",
        "BodyPartExamined": "",
    }

    logger.info(json.dumps(template, indent=2))
    if not segment_attributes:
        logger.error("Missing Attributes/Empty Label provided")
        return None

    if use_itk:
        output_file = itk_image_to_dicom_seg(label, series_dir, template)
    else:
        template = pydicom_seg.template.from_dcmqi_metainfo(template)
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,
            skip_empty_slices=False,
            skip_missing_segment=False,
        )

        # Read source Images
        series_dir = pathlib.Path(series_dir)
        image_files = series_dir.glob(file_ext)
        image_datasets = [dcmread(str(f), stop_before_pixels=True) for f in image_files]
        logger.info(f"Total Source Images: {len(image_datasets)}")

        mask = SimpleITK.ReadImage(label)
        mask = SimpleITK.Cast(mask, SimpleITK.sitkUInt16)

        output_file = tempfile.NamedTemporaryFile(suffix=".dcm").name
        dcm = writer.write(mask, image_datasets)
        dcm.save_as(output_file)

    logger.info(f"nifti_to_dicom_seg latency : {time.time() - start} (sec)")
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


def dicom_seg_to_itk_image(label, output_ext=".seg.nrrd"):
    filename = label if not os.path.isdir(label) else os.path.join(label, os.listdir(label)[0])

    dcm = pydicom.dcmread(filename)
    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm)
    image = result.image

    output_file = tempfile.NamedTemporaryFile(suffix=output_ext).name

    SimpleITK.WriteImage(image, output_file, True)

    if not os.path.exists(output_file):
        logger.warning(f"Failed to convert DICOM-SEG {label} to ITK image")
        return None

    logger.info(f"Result/Output File: {output_file}")
    return output_file
