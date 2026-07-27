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
from datetime import datetime

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

    def test_save_label_preserves_seg_nrrd_suffix(self):
        with tempfile.TemporaryDirectory() as studies:
            image_file = os.path.join(studies, "case-001.nrrd")
            label_dir = os.path.join(studies, "labels", DefaultLabelTag.FINAL)
            source_label_file = os.path.join(studies, "case-001.seg.nrrd")

            os.makedirs(label_dir, exist_ok=True)
            open(image_file, "wb").close()
            open(source_label_file, "wb").close()

            datastore = LocalDatastore(studies, extensions=["*.nrrd"], auto_reload=False)
            datastore.save_label("case-001", source_label_file, DefaultLabelTag.FINAL, {"status": "approved"})

            self.assertTrue(datastore.get_label_uri("case-001", DefaultLabelTag.FINAL).endswith("case-001.seg.nrrd"))
            self.assertEqual(datastore.get_label_info("case-001", DefaultLabelTag.FINAL).get("name"), "case-001.seg.nrrd")

    def test_non_seg_nrrd_label_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as studies:
            datastore = LocalDatastore(studies, extensions=["*.nii.gz"], auto_reload=False)

            self.assertEqual(datastore._to_label_id("case-001.seg.nii.gz"), ("case-001.seg", ".nii.gz"))

    def test_update_label_info_sets_last_reviewed(self):
        with tempfile.TemporaryDirectory() as studies:
            image_file = os.path.join(studies, "case-001.nrrd")
            source_label_file = os.path.join(studies, "case-001.seg.nrrd")

            open(image_file, "wb").close()
            open(source_label_file, "wb").close()

            datastore = LocalDatastore(studies, extensions=["*.nrrd"], auto_reload=False)
            datastore.save_label("case-001", source_label_file, DefaultLabelTag.FINAL, {})
            datastore.update_label_info("case-001", DefaultLabelTag.FINAL, {"status": "approved"})

            last_reviewed = datastore.get_label_info("case-001", DefaultLabelTag.FINAL).get("last_reviewed")
            self.assertIsNotNone(last_reviewed)
            datetime.fromisoformat(last_reviewed)

    def test_update_label_info_preserves_explicit_last_reviewed(self):
        with tempfile.TemporaryDirectory() as studies:
            image_file = os.path.join(studies, "case-001.nrrd")
            source_label_file = os.path.join(studies, "case-001.seg.nrrd")
            explicit_last_reviewed = "2024-01-15T10:30:00"

            open(image_file, "wb").close()
            open(source_label_file, "wb").close()

            datastore = LocalDatastore(studies, extensions=["*.nrrd"], auto_reload=False)
            datastore.save_label("case-001", source_label_file, DefaultLabelTag.FINAL, {})
            datastore.update_label_info(
                "case-001",
                DefaultLabelTag.FINAL,
                {"status": "approved", "last_reviewed": explicit_last_reviewed},
            )

            self.assertEqual(
                datastore.get_label_info("case-001", DefaultLabelTag.FINAL).get("last_reviewed"),
                explicit_last_reviewed,
            )

    def test_update_label_info_falls_back_to_existing_ts_for_legacy_reviews(self):
        with tempfile.TemporaryDirectory() as studies:
            image_file = os.path.join(studies, "case-001.nrrd")
            source_label_file = os.path.join(studies, "case-001.seg.nrrd")
            legacy_ts = 1705314600

            open(image_file, "wb").close()
            open(source_label_file, "wb").close()

            datastore = LocalDatastore(studies, extensions=["*.nrrd"], auto_reload=False)
            datastore.save_label("case-001", source_label_file, DefaultLabelTag.FINAL, {})

            label_info = datastore.get_label_info("case-001", DefaultLabelTag.FINAL)
            label_info.pop("last_reviewed", None)
            label_info.update({"status": "approved", "ts": legacy_ts})

            datastore.update_label_info("case-001", DefaultLabelTag.FINAL, {"comment": "legacy review"})

            self.assertEqual(
                datastore.get_label_info("case-001", DefaultLabelTag.FINAL).get("last_reviewed"),
                datetime.fromtimestamp(legacy_ts).isoformat(),
            )


if __name__ == "__main__":
    unittest.main()
