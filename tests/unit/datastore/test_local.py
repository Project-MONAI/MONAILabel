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
import tempfile
import unittest

from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import DefaultLabelTag


class TestLocalDatastore(unittest.TestCase):
    def test_seg_nrrd_label_matches_image_id(self):
        with tempfile.TemporaryDirectory() as studies:
            image_file = os.path.join(studies, "case-001.nrrd")
            label_dir = os.path.join(studies, "labels", DefaultLabelTag.FINAL)
            label_file = os.path.join(label_dir, "case-001.seg.nrrd")

            os.makedirs(label_dir, exist_ok=True)
            open(image_file, "wb").close()
            open(label_file, "wb").close()

            datastore = LocalDatastore(studies, extensions=["*.nrrd"], auto_reload=False)

            self.assertEqual(datastore.get_labels_by_image_id("case-001"), {DefaultLabelTag.FINAL: "case-001"})
            self.assertTrue(datastore.get_label_uri("case-001", DefaultLabelTag.FINAL).endswith("case-001.seg.nrrd"))

    def test_non_seg_nrrd_label_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as studies:
            datastore = LocalDatastore(studies, extensions=["*.nii.gz"], auto_reload=False)

            self.assertEqual(datastore._to_label_id("case-001.seg.nii.gz"), ("case-001.seg", ".nii.gz"))


if __name__ == "__main__":
    unittest.main()