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
import shutil
import tempfile
import unittest

from monailabel.datastore.local import LocalDatastore


class TestLocalDatastorePathConfinement(unittest.TestCase):
    """Image/label IDs and label tags must resolve inside the datastore image/labels directories."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="monailabel_ds_confine_")
        self.studies = os.path.join(self.base, "studies")
        os.makedirs(self.studies, exist_ok=True)
        self.datastore = LocalDatastore(self.studies)

        self.image_file = os.path.join(self.base, "image.nii.gz")
        with open(self.image_file, "wb") as f:
            f.write(b"image")

        self.image_id = "image_1"
        self.datastore.add_image(self.image_id, self.image_file, {})

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_add_image_rejects_id_outside_image_dir(self):
        outside = os.path.realpath(os.path.join(self.base, "other.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.add_image("../../other", self.image_file, {})
        self.assertFalse(os.path.exists(outside))

    def test_add_image_accepts_plain_id(self):
        image_id = self.datastore.add_image("image_2", self.image_file, {})
        self.assertEqual(image_id, "image_2")
        self.assertTrue(os.path.exists(os.path.join(self.studies, "image_2.nii.gz")))

    def test_save_label_rejects_tag_outside_labels_dir(self):
        outside = os.path.realpath(os.path.join(self.base, "labels", "final", "other.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.save_label(self.image_id, self.image_file, "../../labels/final/other", {})
        self.assertFalse(os.path.exists(outside))

    def test_save_label_rejects_image_id_outside_labels_dir(self):
        # an existing datastore entry whose ID does not resolve inside the labels directory
        self.datastore._datastore.objects["../../other"] = self.datastore._datastore.objects[self.image_id]
        outside = os.path.realpath(os.path.join(self.base, "other.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.save_label("../../other", self.image_file, "final", {})
        self.assertFalse(os.path.exists(outside))

    def test_save_label_accepts_plain_input(self):
        label_id = self.datastore.save_label(self.image_id, self.image_file, "final", {})
        self.assertEqual(label_id, self.image_id)
        self.assertTrue(os.path.exists(os.path.join(self.studies, "labels", "final", "image_1.nii.gz")))


if __name__ == "__main__":
    unittest.main()
