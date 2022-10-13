"""
Read selected tags from the first DICOM file in each subdirectory and save as CSV.

Specify dicom_tags to retrieve using keywords, e.g. Series Description (0008,103E) is 'SeriesDescription'.
https://dicom.innolitics.com/ciods/mr-image/general-series/0008103e

"""

import csv
import timeit
from pathlib import Path

import pandas as pd
import pydicom

root_dir = Path(
    "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeurosurgicalAtlas/DrTures/dicom-files/"
)
di_out = Path(
    "/home/andres/Documents/workspace/disk-workspace/MONAILabel/sample-apps/radiology/series_selection/heuristic_approach/"
)

fields = [
    "ImagesInAcquisition",
    "Modality",
    "ImageType",
    "AngioFlag",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "BodyPartExamined",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "StudyDescription",
    "StudyInstanceUID",
    "PatientAge",
    "SeriesDate",
    "SeriesTime",
    "SeriesDescription",
    "ProtocolName",
    "SeriesInstanceUID",
    "SeriesNumber",
    "ManufacturerModelName",
    "SequenceName",
    "MagneticFieldStrength",
    "ReceiveCoilName",
    "FlipAngle",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "DiffusionBValue",
]

dicom_dirs = root_dir.glob("*/*/*")

metadata = []
count_all = 0
count_dcm = 0
print(f"Constructing DICOM index from {dicom_dirs}")
tic = timeit.default_timer()
for subdirectory in dicom_dirs:

    print(f"Reading dicom folder: {subdirectory}")

    count_all += 1
    dcm_files = list(subdirectory.glob("*.dcm"))

    # Discard by number of Slices
    if len(dcm_files) == 0:
        continue

    # or len(dcm_files) > 300

    # Load the first .dcm file in the directory
    d = pydicom.dcmread(dcm_files[0])

    # # Discard by Modality
    # if d['Modality'].value != 'MR':
    #     continue
    #
    # # Discard by Image type
    # i_type = list(d['ImageType'].value)
    # if i_type:
    #     if 'DERIVED' in i_type:
    #         continue
    # else:
    #     continue

    count_dcm += 1

    # Read selected fields into dictionary
    this_data = {"dcm_file": str(str(dcm_files[0].parent.absolute())[112:])}

    for field in fields:
        if field == "ImageType":
            this_data[field] = list(d[field].value) if field in d else None
        elif field == "ImagesInAcquisition":
            this_data[field] = len(dcm_files)
        else:
            this_data[field] = d[field].value if field in d else None
    metadata.append(this_data)

print("\r", end="")
toc = timeit.default_timer()
run_time = toc - tic
print(
    f"Indexed {count_all} directories, finding .dcm files in {count_dcm}, giving a final index of {len(metadata)} entries in {run_time:.2f} seconds."
)
print("Converting to dataframe")
dicom_index = pd.DataFrame.from_dict(metadata)
print("Saving as CSV")
dicom_index.to_csv(di_out.with_suffix(".csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
