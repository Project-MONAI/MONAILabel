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

from monailabel.datastore.utils.colors import GENERIC_ANATOMY_COLORS, get_segment_color

# Number of unnamed segments used to check that fallback colors are distinct.
_UNNAMED_SEGMENTS = 5
# The color every unnamed segment used to collapse to before the fix (issue #1751).
_LEGACY_RED = [255, 0, 0]
# Black background color that must never be assigned to a segment.
_BLACK = [0, 0, 0]


class TestGetSegmentColor(unittest.TestCase):
    def test_explicit_color_takes_precedence(self):
        self.assertEqual(get_segment_color("anything", {"color": [10, 20, 30]}, 0), [10, 20, 30])

    def test_explicit_color_is_truncated_to_rgb(self):
        # An RGBA (or longer) color is reduced to its first three channels.
        self.assertEqual(get_segment_color("anything", {"color": [10, 20, 30, 255]}, 0), [10, 20, 30])

    def test_known_anatomy_name_uses_palette(self):
        self.assertEqual(get_segment_color("bone"), list(GENERIC_ANATOMY_COLORS["bone"]))

    def test_unnamed_segments_get_distinct_colors(self):
        # Regression for #1751: segments without an explicit color and without a
        # known anatomy name previously all fell back to red, so a multi-segment
        # DICOM SEG showed up entirely red. They must now be distinct.
        colors = [get_segment_color(f"Segment_{i}", {}, i) for i in range(_UNNAMED_SEGMENTS)]
        distinct = {tuple(color) for color in colors}
        self.assertEqual(len(distinct), _UNNAMED_SEGMENTS)
        self.assertNotIn(_LEGACY_RED, colors)

    def test_background_is_never_selected(self):
        # The black background entry must never be handed to a real segment.
        for i in range(len(GENERIC_ANATOMY_COLORS)):
            self.assertNotEqual(get_segment_color("no-such-name", {}, i), _BLACK)


if __name__ == "__main__":
    unittest.main()
