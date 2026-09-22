"""Prompt evaluation must compare effective targets without accepting wrong context."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from monailabel.core.models import AssistantContext
from monailabel.server.assistant_tools.learning import EvaluateArgs, evaluate


@pytest.mark.parametrize("reference", ["evaluation_set_id", "snapshot_id"])
def test_selected_comparison_needs_no_copied_identifiers(reference):
    requests = []

    def start(project_id, request, **kwargs):
        requests.append(request)
        return SimpleNamespace(id="evaluation-job")

    context = AssistantContext(
        model_id="selected-candidate", baseline_id="selected-baseline", **{reference: "held-out"}
    )
    tools = SimpleNamespace(
        context=context,
        project=SimpleNamespace(id="project"),
        user=SimpleNamespace(id="owner"),
        service=SimpleNamespace(learning=SimpleNamespace(evaluate=start)),
    )
    assert evaluate(tools, EvaluateArgs()).job_id == "evaluation-job"
    (request,) = requests
    assert request.candidate_id == "selected-candidate"
    assert request.baseline_id == "selected-baseline"
    assert getattr(request, reference) == "held-out"


@pytest.mark.parametrize(
    "arguments,context,expected",
    [
        ({}, {"learner_id": "snare"}, "Snare U-Net"),
        ({}, {"learner_id": "nuclei"}, "Nuclei U-Net"),
        ({"learner_name": "Snare U-Net"}, {"learner_id": "nuclei"}, "Snare U-Net"),
        ({"learner_id": "snare"}, {"learner_id": "nuclei"}, "Snare U-Net"),
        ({"mode": "continue"}, {"model_id": "checkpoint"}, "Snare U-Net"),
        ({}, {}, None),
        ({}, {"learner_id": "unknown"}, None),
    ],
)
def test_evaluation_resolves_selected_or_named_learner(arguments, context, expected):
    root = Path(__file__).resolve().parents[1]
    recipe = runpy.run_path(str(root / "examples/evaluate_coordinators.py"))
    data = {
        "context": context,
        "models": [{"id": "checkpoint", "learner_id": "snare"}],
        "learners": [
            {"id": "snare", "name": "Snare U-Net"},
            {"id": "nuclei", "name": "Nuclei U-Net"},
        ],
    }
    assert recipe["effective_learner_name"](arguments, data) == expected


def test_comparison_benchmark_selects_distinct_trained_versions():
    root = Path(__file__).resolve().parents[1]
    recipe = runpy.run_path(str(root / "examples/evaluate_coordinators.py"))
    data = recipe["metadata"]("web", "radiology", "compare_versions")
    context = data["context"]
    models = {model["id"]: model for model in data["models"]}
    assert context["model_id"] != context["baseline_id"]
    assert models[context["model_id"]]["trained"]
    assert models[context["baseline_id"]]["trained"]
