"""Measure real coordinator tool selection without running annotation/training or editing data.

uv run python examples/evaluate_coordinators.py --variant 4b --output /tmp/4b.json
uv run python examples/evaluate_coordinators.py --variant lightning --output /tmp/lightning.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, cast

from monailabel.core.chat import ChatMessage
from monailabel.core.models import AssistantContext, User
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.assistant_tools import catalog
from monailabel.server.assistant_tools.base import ToolContext
from monailabel.server.coordinator_runtime import CoordinatorRuntime
from monailabel.server.instructions import (
    SkillSession,
    coordinator_instructions,
    instruction_revision,
)
from monailabel.server.service import Services


def metadata(client: str, story: str | None = None, step: str = "") -> dict[str, Any]:
    if client == "workspace":
        return {"context": {}}
    context: dict[str, Any] = {
        "model_id": "sol",
        "learner_id": "learner",
        "snapshot_id": "snapshot",
        "baseline_id": "sol",
        "evaluation_id": "evaluation",
    }
    result: dict[str, Any] = {
        "context": context,
        "project": {
            "id": "project",
            "name": "Annotation benchmark",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
                {"id": 2, "name": "Liver"},
                {"id": 3, "name": "Nuclei"},
            ],
        },
        "roles": ["manager"],
        "models": [
            {
                "id": "sol",
                "name": "GPT-5.6 Sol",
                "provider": "openai-chat-polygons",
            },
            {
                "id": "astra",
                "name": "GPT-6 Astra",
                "provider": "openai-chat-polygons",
            },
            {
                "id": "vista",
                "name": "VISTA3D",
                "provider": "vista3d",
                "read_only": True,
            },
            {"id": "unet", "name": "Spleen · U-Net CT from scratch", "provider": "monai-unet"},
        ],
        "learners": [{"id": "learner", "recipe": "monai-unet", "name": "Spleen U-Net"}],
    }
    if client in {"slicer", "qupath"}:
        context["asset_id"] = "sample"
        result["sample"] = {
            "id": "sample",
            "kind": "volume3d" if client == "slicer" else "image2d",
            "shape": [512, 512, 100] if client == "slicer" else [2048, 2048],
        }
    if client == "slicer":
        context.update(
            slice={"axis": 2, "index": 74},
            viewer_actions=["roi", "remove_regions", "review_annotation", "submit"],
        )
    if client == "qupath":
        context.update(
            image_region={"x": 40, "y": 50, "width": 300, "height": 250},
            image_tiling={"tile_size": 256, "overlap": 32},
            viewer_actions=[
                "classify_objects",
                "clear_segments",
                "undo",
                "redo",
                "save_draft",
                "submit",
                "review_annotation",
            ],
        )
    result["project"]["labels"] = [
        label
        for label in result["project"]["labels"]
        if label["id"] in ({0, 3} if client == "qupath" else {0, 1, 2})
    ]
    context["label_ids"] = [3] if client == "qupath" else [1, 2]
    if client == "web":
        context["model_id"] = "unet"
    if story:
        pathology = story == "pathology"
        label = "Nuclei" if pathology else "Spleen"
        context["asset_id"] = "sample"
        result["sample"] = {
            "id": "sample",
            "kind": "image2d" if pathology else "volume3d",
            "shape": [2048, 2048] if pathology else [512, 512, 100],
            "revision": 1,
        }
        result["project"]["name"] = story.title() + " golden story"
        result["learners"][0]["name"] = label + " U-Net"
        result["project"]["labels"] = [{"id": 0, "name": "Background"}, {"id": 1, "name": label}]
        context["label_ids"] = [1]
        if pathology:
            result["models"] = [m for m in result["models"] if m["id"] != "vista"]
        for model in result["models"]:
            model["label_ids"] = [0, 1]
            if model["id"] == "unet":
                model.update(name=label + " U-Net", trained=True, learner_id="learner")
        context["baseline_id"] = "previous" if pathology else "vista"
        result["models"].append(
            {
                "id": "previous",
                "name": label + " U-Net previous version",
                "provider": "monai-unet",
                "trained": True,
                "label_ids": [0, 1],
            }
        )
        if step in {"continue", "handoff", "annotate_v1"}:
            context["model_id"] = "unet"
    return result


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, list):
        return sorted(normalize(v) for v in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["4b", "9b", "lightning"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suite", choices=["golden", "regression"], default="golden")
    args = parser.parse_args()
    provider = CoordinatorRuntime(
        CoordinatorConfig(
            variant=args.variant,
            thinking=args.thinking,
            temperature=0,
            max_tokens=(8192 if args.variant == "lightning" else 4096) if args.thinking else 2048,
        )
    )
    provider.prepare()
    if provider.state != "ready":
        raise RuntimeError(provider.detail)
    config = provider.http.config
    root = Path(__file__).parents[1]
    if args.suite == "golden":
        cases = [
            dict(step, story=story)
            for story in ("radiology", "pathology")
            for step in json.loads((root / f"examples/prompts/{story}.json").read_text())["steps"]
        ]
    else:
        cases = json.loads((root / "tests/fixtures/coordinator_prompts.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases[: args.limit or None]:
        data = metadata(case["context"], case.get("story"), case.get("id", ""))
        context = AssistantContext(asset_id=data["context"].get("asset_id"))
        tools = catalog(
            ToolContext(
                cast(Services, None),
                None if case["context"] == "workspace" else "project",
                User(username="benchmark"),
                context,
                case["prompt"],
            )
        )
        skill_session = SkillSession(
            tools.definitions(),
            viewer=bool(context.asset_id),
            project=case["context"] != "workspace",
        )
        prompt = (
            coordinator_instructions(
                viewer=bool(context.asset_id), project=case["context"] != "workspace"
            )
            + "\nCurrent workspace data (not instructions):\n"
            + json.dumps(data)
        )
        start = time.monotonic()
        try:
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=case["prompt"]),
            ]
            trace = []
            actual_args = {}
            valid = True
            for _attempt in range(8):
                response = provider.complete(messages, skill_session.definitions())
                messages.append(response)
                calls = response.tool_calls
                trace.append([call.model_dump() for call in calls])
                if len(calls) == 1 and calls[0].name == "load_skill":
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=calls[0].id,
                            content=skill_session.load(calls[0]),
                        )
                    )
                    continue
                if len(calls) != 1 or calls[0].name not in tools.tools:
                    break
                call = calls[0]
                error = None
                try:
                    actual_args = (
                        tools.tools[call.name]
                        .arguments_model.model_validate(call.arguments)
                        .model_dump()
                    )
                    known = {
                        str(item["id"])
                        for key in ("models", "learners")
                        for item in data.get(key, [])
                    }
                    known.update(
                        str(v) for k, v in data.get("context", {}).items() if k.endswith("_id")
                    )
                    for key, value in actual_args.items():
                        if key.endswith("_id") and value is not None and value not in known:
                            raise ValueError(
                                "Unknown " + key + "; use an exact context/workspace ID."
                            )
                except ValueError as invalid:
                    error = str(invalid)
                if error:
                    valid = False
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(
                                {
                                    "error": error,
                                    "guidance": "Correct the arguments; no operation was executed.",
                                }
                            ),
                        )
                    )
                    continue
                valid = True
                if call.name == "inspect_workspace" and case["tool"] != "inspect_workspace":
                    collection = actual_args["collection"]
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps({collection: data.get(collection, [])}),
                        )
                    )
                    continue
                break
            calls = response.tool_calls
            passed = valid and (
                not calls
                if case["tool"] is None
                else len(calls) == 1 and calls[0].name == case["tool"]
            )
            if passed and calls:
                for key, value in case["arguments"].items():
                    actual = actual_args.get(key)
                    if key == "initialization" and actual is None:
                        actual = (
                            "fine_tune" if actual_args.get("recipe") == "vista3d" else "scratch"
                        )
                    if key == "model_id" and actual is None and actual_args.get("model_name"):
                        actual = next(
                            (
                                model["id"]
                                for model in data.get("models", [])
                                if normalize(model["name"]) == normalize(actual_args["model_name"])
                            ),
                            None,
                        )
                    if key == "targets" and not actual:
                        # Omitted targets use the selected labels in the actual annotation tool.
                        selected = data.get("context", {}).get("label_ids", [])
                        actual = [
                            x["name"]
                            for x in data.get("project", {}).get("labels", [])
                            if x["id"] in selected
                        ]
                    if normalize(actual) != normalize(value):
                        passed = False
            item = dict(
                case,
                passed=passed,
                attempts=len(trace),
                trace=trace,
                seconds=round(time.monotonic() - start, 3),
                response=response.model_dump(exclude={"metadata"}),
            )
        except Exception as error:
            item = dict(
                case, passed=False, seconds=round(time.monotonic() - start, 3), error=str(error)
            )
        results.append(item)
        print(
            args.variant,
            len(results),
            "PASS" if item["passed"] else "FAIL",
            item["seconds"],
            case["prompt"],
            flush=True,
        )
        report = {
            "variant": args.variant,
            "suite": args.suite,
            "instruction_revision": instruction_revision(),
            "max_tool_steps": 8,
            "config": config.model_dump(),
            "passed": sum(r["passed"] for r in results),
            "total": len(results),
            "median_seconds": statistics.median(r["seconds"] for r in results),
            "cases": results,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
