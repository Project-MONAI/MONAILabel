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

"""The documented prompts drive real service/learning checks independently of LLM inference."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_guide_uses_exact_golden_prompt_definitions():
    module = runpy.run_path(str(ROOT / "examples/render_golden_prompts.py"))
    assert module["render"]() in module["TARGET"].read_text()
    quickstart = runpy.run_path(str(ROOT / "examples/quickstart_prompts.py"))
    assert quickstart["cases"]()


@pytest.mark.parametrize("story", ["radiology", "pathology"])
def test_complete_golden_learning_story(story):
    pytest.importorskip("monailabel.monai")
    module = runpy.run_path(str(ROOT / "examples/verify_golden_stories.py"))
    report = module["run_story"](story)
    assert report["held_out_sources"] == 2
    assert report["parent_checkpoint_unchanged"] and report["defaults_unchanged"]
    assert report["spatial_dims"] == (2 if story == "pathology" else 3)
    assert {s["id"] for s in report["steps"]} >= {
        "project",
        "import",
        "models",
        "submit",
        "accept",
        "train",
        "continue",
        "compare_versions",
    }
