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
        result_file_first_stage, result_json_first_stage = self.model_localization_spine(request)

        # Request for second stage
        second_stage_request = copy.deepcopy(request)
        second_stage_request["first_stage_pred"] = result_file_first_stage

        #################################################
        # Run second stage
        #################################################
        _, result_json_second_stage = self.model_localization_vertebra(second_stage_request)

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
