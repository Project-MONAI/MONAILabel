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
import copy
import logging
import tempfile

from monai.transforms import LoadImaged, SaveImage

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import MergeAllPreds

logger = logging.getLogger(__name__)


class InferVertebraPipeline(InferTask):
    def __init__(
        self,
        model_localization_spine: InferTask,
        model_localization_vertebra: InferTask,
        model_segmentation_vertebra: InferTask,
        type=InferType.SEGMENTATION,
        dimension=3,
        description="Combines three stages for vertebra segmentation",
        output_largest_cc=False,
    ):
        # Should we consider this class as the last infer stage?
        super().__init__(
            path=None,  # THIS SHOULD BE NONE??
            network=None,  # THIS SHOULD BE NONE??
            type=type,
            labels=None,
            dimension=dimension,
            description=description,
        )
        self.model_localization_spine = model_localization_spine
        self.model_localization_vertebra = model_localization_vertebra
        self.model_segmentation_vertebra = model_segmentation_vertebra
        self.output_largest_cc = output_largest_cc

    def post_transforms(self, data=None):
        return None

    def pre_transforms(self, data=None):
        return None

    def is_valid(self) -> bool:
        return True

    def __call__(self, request):

        #################################################
        # Run first stage
        #################################################
        # result_file_first_stage, result_json_first_stage = self.model_localization_spine(request)

        # These outputs are for verse111.nii.gz
        result_file_first_stage = "/tmp/tmpxcy0cnba.nii.gz"
        result_json_first_stage = {
            "label_names": {
                "C1": 1,
                "C2": 2,
                "C3": 3,
                "C4": 4,
                "C5": 5,
                "C6": 6,
                "C7": 7,
                "Th1": 8,
                "Th2": 9,
                "Th3": 10,
                "Th4": 11,
                "Th5": 12,
                "Th6": 13,
                "Th7": 14,
                "Th8": 15,
                "Th9": 16,
                "Th10": 17,
                "Th11": 18,
                "Th12": 19,
                "L1": 20,
                "L2": 21,
                "L3": 22,
                "L4": 23,
                "L5": 24,
            },
            "latencies": {
                "pre": 1.87,
                "infer": 0.5,
                "invert": 0.0,
                "post": 2.46,
                "write": 0.18,
                "total": 5.01,
                "transform": {
                    "pre": {
                        "LoadImaged": 1.339,
                        "EnsureTyped": 0.0055,
                        "EnsureChannelFirstd": 0.0003,
                        "NormalizeIntensityd": 0.0063,
                        "GaussianSmoothd": 0.5045,
                        "ScaleIntensityd": 0.0049,
                    },
                    "post": {
                        "EnsureTyped": 0.0002,
                        "Activationsd": 0.002,
                        "AsDiscreted": 0.0008,
                        "KeepLargestConnectedComponentd": 2.4561,
                        "BinaryMaskd": 0.0003,
                        "Restored": 0.0004,
                    },
                },
            },
        }

        # Request for second stage
        second_stage_request = copy.deepcopy(request)
        second_stage_request["first_stage_pred"] = result_file_first_stage

        #################################################
        # Run second stage
        #################################################
        # _, result_json_second_stage = self.model_localization_vertebra(second_stage_request)

        # These outputs are for verse111.nii.gz
        result_json_second_stage = {
            "centroids": [
                {"label_18": [18, 86, 86, 253]},
                {"label_19": [19, 87, 92, 222]},
                {"label_20": [20, 87, 98, 193]},
                {"label_21": [21, 86, 105, 163]},
                {"label_22": [22, 91, 112, 139]},
                {"label_23": [23, 98, 124, 113]},
                {"label_24": [24, 102, 125, 83]},
            ],
            "label_names": {
                "C1": 1,
                "C2": 2,
                "C3": 3,
                "C4": 4,
                "C5": 5,
                "C6": 6,
                "C7": 7,
                "Th1": 8,
                "Th2": 9,
                "Th3": 10,
                "Th4": 11,
                "Th5": 12,
                "Th6": 13,
                "Th7": 14,
                "Th8": 15,
                "Th9": 16,
                "Th10": 17,
                "Th11": 18,
                "Th12": 19,
                "L1": 20,
                "L2": 21,
                "L3": 22,
                "L4": 23,
                "L5": 24,
            },
            "latencies": {
                "pre": 0.41,
                "infer": 0.41,
                "invert": 0.0,
                "post": 2.56,
                "write": 0.0,
                "total": 3.37,
                "transform": {
                    "pre": {
                        "LoadImaged": 0.3748,
                        "EnsureTyped": 0.0073,
                        "EnsureChannelFirstd": 0.0005,
                        "CropForegroundd": 0.0096,
                        "NormalizeIntensityd": 0.0053,
                        "GaussianSmoothd": 0.0024,
                        "ScaleIntensityd": 0.0059,
                    },
                    "post": {
                        "EnsureTyped": 0.0002,
                        "Activationsd": 0.0012,
                        "AsDiscreted": 0.0007,
                        "KeepLargestConnectedComponentd": 2.1686,
                        "Restored": 0.0004,
                        "VertebraLocalizationSegmentation": 0.3792,
                    },
                },
            },
        }

        #################################################
        # Run third stage
        #################################################
        # Request for third stage
        third_stage_request = copy.deepcopy(second_stage_request)
        all_outs = {}
        all_keys = []
        for centroid in result_json_second_stage["centroids"]:
            third_stage_request["centroids"] = [centroid]
            # TO DO:
            # 1/ Remove the AsDiscrete transform in third stage infer so we get pre-activation outputs
            # 2/ Don't load the volume everytime this performs inference
            result_file, result_json_third_stage = self.model_segmentation_vertebra(third_stage_request)
            all_keys.append(list(centroid.keys())[0])
            all_outs[list(centroid.keys())[0]] = result_file

        # Once all the predictions are obtained, use the label dict to reconstruct the output
        out = LoadImaged(keys=all_keys, reader="ITKReader")(all_outs)
        result_file_third_stage = MergeAllPreds(keys=all_keys)(out)

        output_file = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
        result_file_third_stage.meta["filename_or_obj"] = output_file
        SaveImage(output_postfix="", output_dir="/tmp/", separate_folder=False)(result_file_third_stage)

        return output_file, result_json_third_stage
