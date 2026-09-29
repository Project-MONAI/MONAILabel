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

"""Exact README prompts and isolated workspace metadata for coordinator checks."""

import json
import re
from pathlib import Path

from monailabel.providers.spatial import MODELS as SPATIAL_MODELS
from monailabel.server.dataset_downloads import sources

ROOT = Path(__file__).resolve().parents[1]


def cases():
    stories = json.loads((ROOT / "examples/prompts/quickstart.json").read_text())["stories"]
    blocks = re.findall(r"```text\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    documented = [
        line for block in blocks for line in block.splitlines() if line and not line.startswith("#")
    ]
    result = [
        dict(step, workflow=story, id=f"{story['id']}-{step['id']}")
        for story in stories
        for step in story["steps"]
    ]
    assert documented == [case["prompt"] for case in result], (
        "Update Quickstart prompt checks to match README.md"
    )
    return result


def workspace_data(case, data):
    """State at this prompt's documented step; no operational tool executes here."""
    story = case["workflow"]
    if case["context"] == "workspace":
        return data
    labels = [{"id": 0, "name": "Background"}] + [
        {"id": i, "name": name} for i, name in enumerate(story["labels"], 1)
    ]
    ids = [label["id"] for label in labels]
    data["project"].update(name=story["name"], labels=labels)
    context = data["context"]
    for field in ("learner_id", "snapshot_id", "evaluation_id", "baseline_id"):
        context.pop(field, None)
    if case["context"] == "web":
        context.clear()
    context["label_ids"] = ids[1:]
    context["model_id"] = "vista" if story["modality"] else "astra"
    data["models"] = [
        {
            "id": "astra",
            "name": "GPT-6 Astra",
            "provider": "openai-chat-polygons",
            "label_ids": ids,
        },
        {
            "id": "vista",
            "name": "VISTA3D",
            "provider": "vista3d",
            "read_only": True,
            "label_ids": ids,
        },
    ]
    if story["id"] == "spleen":
        data["models"].extend(
            [
                {
                    "id": "total",
                    "name": "TotalSegmentator CT",
                    "provider": "totalsegmentator-ct",
                    "read_only": True,
                    "label_ids": ids,
                },
                {
                    "id": "nninteractive",
                    "name": "nnInteractive",
                    "provider": "nninteractive",
                    "read_only": True,
                    "label_ids": [0],
                    "spatial_prompts_required": True,
                    "interaction": SPATIAL_MODELS["nninteractive"].interaction.model_dump(),
                },
            ]
        )
        if case["context"] == "slicer":
            context["model_id"] = "nninteractive"
            context["interaction_target"] = "Spleen"
            context["interaction_mode"] = "positive"
            context["viewer_actions"] = list(
                set(context.get("viewer_actions", []))
                | {"edit_spatial_prompts", "set_interaction_mode", "submit"}
            )
            plane = context.get("slice") or {"axis": 2, "index": 4, "window": [0, 100]}
            context["slice"] = plane
            point = [6.0, 6.0, 6.0]
            point[plane["axis"]] = plane["index"]
            context["spatial_objects"] = [
                {
                    "id": "spleen-point",
                    "target": "Spleen",
                    "kind": "point",
                    "coordinates": [point],
                    "positive": True,
                    "selected": False,
                }
            ]
    data["learners"] = []
    step = case["id"].removeprefix(story["id"] + "-")
    if step in {"train", "compare", "results", "predict"}:
        data["learners"] = [
            {"id": "learner", "name": story["learner"], "recipe": story["recipe"], "label_ids": ids}
        ]
        context["learner_id"] = "learner"
    if step in {"compare", "results", "predict"}:
        data["models"].append(
            {
                "id": "trained",
                "name": story["learner"],
                "provider": story["recipe"],
                "trained": True,
                "learner_id": "learner",
                "label_ids": ids,
            }
        )
        context["model_id"] = "trained"
        if story["modality"]:
            context["baseline_id"] = "vista"
    data["evaluation_sets"] = (
        [{"id": "fixed", "name": "Fixed evaluation set", "archived": False}]
        if story["modality"]
        else []
    )
    if data["evaluation_sets"]:
        context["evaluation_set_id"] = "fixed"
    data["dataset_templates"] = [
        source.model_dump(
            mode="json",
            include={
                "id",
                "name",
                "category",
                "kind",
                "importable",
                "has_masks",
                "channels",
                "sections",
            },
        )
        for source in sources()
    ]
    data["evaluations"] = []
    data["recent_jobs"] = []
    return data
