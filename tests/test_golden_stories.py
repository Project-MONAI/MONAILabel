"""The documented prompts drive real service/learning checks independently of LLM inference."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_guide_uses_exact_golden_prompt_definitions():
    module = runpy.run_path(str(ROOT / "examples/render_golden_prompts.py"))
    assert module["render"]() in (ROOT / "README.md").read_text()


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
