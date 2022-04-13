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

import os
import unittest

from monailabel.datastore.local import LocalDatastore
from monailabel.utils.others.planner import HeuristicPlanner


class TestPlanner(unittest.TestCase):
    def run_planner(self):

        base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        data_dir = os.path.join(base_dir, "tests", "data")
        studies = os.path.join(data_dir, "dataset", "local", "spleen")

        datastore = LocalDatastore(
            studies,
            extensions=["*.nii.gz", "*.nii"],
            auto_reload=True,
        )
        planner_object = HeuristicPlanner()
        planner_object.run(datastore=datastore)

        spatial_size = planner_object.spatial_size
        target_spacing = planner_object.target_spacing

        # Image stats for intensity normalization
        max_pix = planner_object.max_pix
        min_pix = planner_object.min_pix
        mean_pix = planner_object.mean_pix
        std_pix = planner_object.std_pix

        self.assertIsNotNone(spatial_size)
        self.assertIsNotNone(target_spacing)
        self.assertGreaterEqual(max_pix, 0)
        self.assertLessEqual(min_pix, 0)
        self.assertLessEqual(mean_pix, 0)
        self.assertGreaterEqual(std_pix, 0)

    def test_planner(self):
        self.run_planner()


if __name__ == "__main__":
    unittest.main()
