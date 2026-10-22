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
from unittest import mock

from monailabel.utils.others.generic import gpu_memory_map

GIB = 1024 * 1024 * 1024


def _which(available):
    def f(cmd):
        return f"/usr/bin/{cmd}" if cmd == available else None

    return f


class TestGpuMemoryMap(unittest.TestCase):
    def test_nvidia_smi(self):
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("nvidia-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", return_value="1000\n2000\n"
        ):
            self.assertEqual(gpu_memory_map(), {0: 1000, 1: 2000})

    def test_nvidia_smi_failure_falls_through_to_rocm(self):
        # A failing nvidia-smi must not block the rocm-smi fallback.
        csv = f"device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,{2 * GIB},{1 * GIB}\n"

        def _which_both(cmd):
            return f"/usr/bin/{cmd}" if cmd in ("nvidia-smi", "rocm-smi") else None

        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which_both), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", side_effect=[OSError("boom"), csv]
        ):
            self.assertEqual(gpu_memory_map(), {0: 1024})

    def test_rocm_smi(self):
        # rocm-smi reports total/used VRAM in bytes; free is derived as total - used.
        csv = f"device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,{2 * GIB},{1 * GIB}\n"
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("rocm-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", return_value=csv
        ):
            self.assertEqual(gpu_memory_map(), {0: 1024})

    def test_rocm_smi_multi_card_skips_warning_lines(self):
        # rocm-smi prints warnings to stdout ahead of the CSV header; both must be skipped.
        csv = (
            "WARNING: AMD GPU device(s) is/are in a low-power state.\n"
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            f"card0,{2 * GIB},{1 * GIB}\n"
            f"card1,{4 * GIB},{1 * GIB}\n"
        )
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("rocm-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", return_value=csv
        ):
            self.assertEqual(gpu_memory_map(), {0: 1024, 1: 3072})

    def test_rocm_smi_failure_returns_default(self):
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("rocm-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", side_effect=OSError("boom")
        ):
            self.assertEqual(gpu_memory_map(), {0: 4300})

    def test_rocm_smi_unparsable_output_returns_default(self):
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("rocm-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", return_value="no cards here\n"
        ):
            self.assertEqual(gpu_memory_map(), {0: 4300})

    def test_rocm_smi_bad_row_preserves_card_index(self):
        # card0 has malformed columns; card1 is valid. card1's memory must map to key 1,
        # and card0 must be absent entirely (not stored as zero free memory).
        smi_output = (
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            "card0,bad,data\n"
            f"card1,{4 * GIB},{1 * GIB}\n"
        )
        with mock.patch("monailabel.utils.others.generic.shutil.which", side_effect=_which("rocm-smi")), mock.patch(
            "monailabel.utils.others.generic.subprocess.check_output", return_value=smi_output
        ):
            result = gpu_memory_map()
            self.assertNotIn(0, result)
            self.assertEqual(result, {1: 3072})

    def test_no_smi_tool_returns_default(self):
        with mock.patch("monailabel.utils.others.generic.shutil.which", return_value=None):
            self.assertEqual(gpu_memory_map(), {0: 4300})


if __name__ == "__main__":
    unittest.main()
