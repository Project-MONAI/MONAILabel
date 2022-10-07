"""
Read selected tags from the first DICOM file in each subdirectory and save as CSV and parquet.

Saves the index dataframe as parquet and CSV.
Specify dicom_tags to retrieve using keywords, e.g. Series Description (0008,103E) is 'SeriesDescription'.
https://dicom.innolitics.com/ciods/mr-image/general-series/0008103e
Requires pyarrow to save parquet files.
"""

import csv
import timeit
from pathlib import Path

import pandas as pd
import pydicom

root_dir = Path(
    "/media/andres/SharedUbuntuWindows/Datasets/radiology/brain/NeuroAtlas-Labels/DrTures/raw-dicom/Patient 064/9395438/"
)
dicom_dir = root_dir
di_out = root_dir / "metadata" / "dicom_index"
fields = [
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "StudyDescription",
    "StudyInstanceUID",
    "PatientAge",
    "SeriesDate",
    "SeriesTime",
    "Modality",
    "SeriesDescription",
    "BodyPartExamined",
    "ProtocolName",
    "SeriesInstanceUID",
    "SeriesNumber",
    "ManufacturerModelName",
    "ImageType",
    "SequenceName",
    "AngioFlag",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "MagneticFieldStrength",
    "ReceiveCoilName",
    "FlipAngle",
    "ContrastBolusAgent",
    "ContrastBolusRoute",
    "DiffusionBValue",
]

metadata = []
count_all = 0
count_dcm = 0
spinner = ["|", "/", "-", "\\"]
print(f"Constructing DICOM index from {dicom_dir}")
tic = timeit.default_timer()
for subdirectory in dicom_dir.glob("**/"):
    print(f"\r{spinner[count_dcm%4]}{spinner[count_all%4]}", end="")
    count_all += 1
    dcm_files = list(subdirectory.glob("*.dcm"))
    if len(dcm_files) == 0:
        continue
    count_dcm += 1

    # Load the first .dcm file in the directory
    d = pydicom.dcmread(dcm_files[0])

    # Read selected fields into dictionary
    this_data = {"dcm_file": str(dcm_files[0])}

    for field in fields:
        if field == "ImageType":
            this_data[field] = list(d[field].value) if field in d else None
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
print("Saving as parquet")
try:
    dicom_index.to_parquet(di_out.with_suffix(".parquet"))
except Exception as e:
    print("Could not save as parquet. Saving as pickle. It's probably a data type pyarrow doesn't like.")
    dicom_index.to_pickle(str(di_out.with_suffix(".pickle")))
    print("Here's the error message:")
    print(repr(e))
print("Done\n")
