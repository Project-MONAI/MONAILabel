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

import unittest
from datetime import datetime, timezone

from monailabel.endpoints import datastore_review
from monailabel.interfaces.exception import LabelNotFoundException


class _MissingLabelDatastore:
    def get_label_info(self, image_id, tag):
        raise LabelNotFoundException(image_id)


class _BrokenDatastore:
    def get_label_info(self, image_id, tag):
        raise RuntimeError("boom")


class TestDatastoreReview(unittest.TestCase):
    def test_safe_label_info_returns_empty_for_missing_label(self):
        info = datastore_review._safe_label_info(_MissingLabelDatastore(), "case-001", "final")

        self.assertEqual(info, {})

    def test_safe_label_info_propagates_unexpected_errors(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            datastore_review._safe_label_info(_BrokenDatastore(), "case-001", "final")

    def test_matches_date_range_handles_naive_value_with_aware_bounds(self):
        parsed = (
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 31, 23, 59, tzinfo=timezone.utc),
        )

        self.assertTrue(datastore_review._matches_date_range("2024-01-15T12:00:00", parsed))

    def test_matches_date_range_handles_aware_value_with_naive_bounds(self):
        parsed = (
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 31, 23, 59),
        )

        self.assertTrue(datastore_review._matches_date_range("2024-01-15T12:00:00+00:00", parsed))


if __name__ == "__main__":
    unittest.main()
