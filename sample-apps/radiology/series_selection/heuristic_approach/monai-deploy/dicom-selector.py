from json import loads as json_loads
from pathlib import Path

from monai.deploy.operators import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

# This path should be given at the patient level
data_path = Path(
    "/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/NeurosurgicalAtlas/DrTures/dicom-files/small-test/"
)


di_out = Path(
    "/home/andres/Documents/workspace/disk-workspace/MONAILabel/sample-apps/radiology/series_selection/heuristic_approach/"
)


# Adding ? after the qualifier makes it perform the match in non-greedy or minimal fashion; as few characters as possible will be matched.
# https://docs.python.org/3/library/re.html
rulesText = """
{
    "selections": [
        {
            "name": "T1",
            "conditions": {
                "Modality": "(?i)MR",
                "SeriesDescription": "((.*)T1_MPRAGE|(.*)T1_BRAVO)"
            }
        },
        {
            "name": "T2",
            "conditions": {
                "Modality": "(?i)MR",
                "SeriesDescription": "((.*)PROP|(.*)PROPELLER)"
            }
        },
        {
            "name": "T1C",
            "conditions": {
                "Modality": "(?i)MR",
                "SeriesDescription": "(.*)AKS_NAVIGATOR"
            }
        },
        {
            "name": "FLAIR",
            "conditions": {
                "Modality": "(?i)MR",
                "SeriesDescription": "(.*)Ax_T2"
            }
        }
    ]
}
"""

# "ImagesInAcquisition": "([0-3]\d{2,})", # There is not support for greater or equal in MONAI Deploy
# We may want to add greater or equal than for "ImagesInAcquisition" tag here:
# https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/monai/deploy/operators/dicom_series_selector_operator.py#L239-L244

# "ImageType": ["ORIGINAL", "PRIMARY"]

loader = DICOMDataLoaderOperator()

study_list = loader.load_data_to_studies(data_path.absolute())

selector = DICOMSeriesSelectorOperator()

sample_selection_rule = json_loads(rulesText)

print(f"Selection rules in JSON:\n{sample_selection_rule}")

# Select only the first series that matches the conditions per name, list of one
# https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/monai/deploy/operators/dicom_series_selector_operator.py#L155
study_selected_series_list = selector.filter(sample_selection_rule, study_list)


print(f"This is the result: {study_selected_series_list}")
