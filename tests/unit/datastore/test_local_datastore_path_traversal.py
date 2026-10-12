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


class TestLocalDatastorePathTraversal(unittest.TestCase):
    """Regression tests for GHSA-cxx7-378q-5622.

    User-supplied image/label IDs and label tags must never be able to write
    files outside the datastore image/labels directories.
    """

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="monailabel_ds_traversal_")
        self.studies = os.path.join(self.base, "studies")
        os.makedirs(self.studies, exist_ok=True)
        self.datastore = LocalDatastore(self.studies)

        self.marker = b"ARBITRARY_FILE_WRITE_VIA_PATH_TRAVERSAL"
        self.image_payload = os.path.join(self.base, "payload.nii.gz")
        with open(self.image_payload, "wb") as f:
            f.write(self.marker)

        self.legit_id = "legit"
        self.datastore.add_image(self.legit_id, self.image_payload, {})

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_add_image_rejects_traversal_id(self):
        outside = os.path.realpath(os.path.join(self.base, "poc_traversal_proof.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.add_image("../../poc_traversal_proof", self.image_payload, {})
        self.assertFalse(os.path.exists(outside))

    def test_add_image_accepts_legitimate_id(self):
        image_id = self.datastore.add_image("good_image", self.image_payload, {})
        self.assertEqual(image_id, "good_image")
        self.assertTrue(os.path.exists(os.path.join(self.studies, "good_image.nii.gz")))

    def test_save_label_rejects_traversal_tag(self):
        outside = os.path.realpath(os.path.join(self.base, "labels", "final", "escaped.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.save_label(self.legit_id, self.image_payload, "../../labels/final/escaped", {})
        self.assertFalse(os.path.exists(outside))

    def test_save_label_rejects_traversal_image_id(self):
        # emulate a datastore already poisoned with a traversal object
        self.datastore._datastore.objects["../../poison"] = self.datastore._datastore.objects[self.legit_id]
        outside = os.path.realpath(os.path.join(self.base, "poison.nii.gz"))
        with self.assertRaises(ValueError):
            self.datastore.save_label("../../poison", self.image_payload, "final", {})
        self.assertFalse(os.path.exists(outside))

    def test_save_label_accepts_legitimate_input(self):
        label_id = self.datastore.save_label(self.legit_id, self.image_payload, "final", {})
        self.assertEqual(label_id, self.legit_id)
        self.assertTrue(os.path.exists(os.path.join(self.studies, "labels", "final", "legit.nii.gz")))


if __name__ == "__main__":
    unittest.main()
