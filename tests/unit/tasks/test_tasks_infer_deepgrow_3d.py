# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import torch
from monai.utils import set_determinism

from monailabel.datastore.local import LocalDatastore
from monailabel.tasks.infer.deepgrow_3d import InferDeepgrow3D


class TestInferDeepgrow2D(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_infer_deepgrow_2d(self):

        base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        data_dir = os.path.join(base_dir, "tests", "data")
        studies = os.path.join(data_dir, "dataset", "local", "spleen")

        datastore = LocalDatastore(
            studies,
            extensions=["*.nii.gz", "*.nii"],
            auto_reload=True,
        )

        data_json = datastore.json()["objects"]

        # Get Data file name
        for key, _value in data_json.items():
            t1 = key
            break

        file_name = data_json[t1]["image"]["info"]["name"]
        file_path = os.path.join(studies, file_name)

        model = torch.nn.Identity(20)
        input = torch.randn(30, 30)
        output = model(input)

        deepgrow_3d_infer = InferDeepgrow3D(network=model, path=file_path)
        pre_transform_list = deepgrow_3d_infer.pre_transforms(None)
        post_transform_list = deepgrow_3d_infer.post_transforms(None)
        deepgrow_inferer = deepgrow_3d_infer.inferer(None)
        deepgrow_inferer_output = deepgrow_inferer(inputs=input, network=model)

        self.assertEqual(output.shape, deepgrow_inferer_output.shape)
        self.assertEqual(len(pre_transform_list), 10)
        self.assertEqual(len(post_transform_list), 5)


if __name__ == "__main__":
    unittest.main()
