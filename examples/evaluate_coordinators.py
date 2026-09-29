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
from monailabel.core.errors import DomainError
from monailabel.core.models import AssistantContext, User
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.spatial import MODELS as SPATIAL_MODELS
from monailabel.server.assistant_tools import catalog
from monailabel.server.assistant_tools.base import ToolContext
from monailabel.server.assistant_tools.workspace import native_viewer_command
from monailabel.server.coordinator_runtime import CoordinatorRuntime
from monailabel.server.instructions import (
    WORKFLOW_TOOL_REMINDER,
    SkillSession,
    coordinator_instructions,
    instruction_revision,
)
from monailabel.server.service import Services


def metadata(client: str, story: str | None = None, step: str = "") -> dict[str, Any]:
    if client == "workspace":
        return {"context": {}}
    context: dict[str, Any] = {
        "model_id": "astra",
        "learner_id": "learner",
        "snapshot_id": "snapshot",
        "baseline_id": "astra",
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
    if client in {"slicer", "ohif", "qupath"}:
        context["asset_id"] = "sample"
        result["sample"] = {
            "id": "sample",
            "kind": "image2d" if client == "qupath" else "volume3d",
            "shape": [2048, 2048] if client == "qupath" else [512, 512, 100],
        }
    if client in {"slicer", "ohif"}:
        result["models"].extend(
            {
                "id": key,
                "name": spec.name,
                "provider": key,
                "read_only": True,
                "interaction": spec.interaction.model_dump(),
            }
            for key, spec in SPATIAL_MODELS.items()
        )
        context.update(
            base_revision=0,
            slice={"axis": 2, "index": 74},
            viewer_actions=[
                "roi",
                "remove_regions",
                "clear_segments",
                "undo",
                "redo",
                "review_annotation",
                "submit",
                "set_interaction_mode",
                "edit_spatial_prompts",
            ],
        )
    if client == "qupath":
        context.update(
            base_revision=0,
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
                "set_interaction_mode",
                "edit_spatial_prompts",
            ],
        )
    result["project"]["labels"] = [
        label
        for label in result["project"]["labels"]
        if label["id"] in ({0, 3} if client == "qupath" else {0, 1, 2})
    ]
    context["label_ids"] = [3] if client == "qupath" else [1, 2]
    if client == "cvat":
        context.update(
            video={
                "video_id": "video",
                "editor_id": "editor",
                "frame": 3,
                "draft_signature": "a" * 64,
            },
            base_revision=0,
            viewer_actions=[
                "clear_video_annotations",
                "undo",
                "redo",
                "save_draft",
                "submit",
                "set_interaction_mode",
                "clear_video_inputs",
            ],
            label_ids=[1],
        )
        result["viewer"] = "cvat"
        result["video"] = {"id": "video", "frames": 40, "width": 320, "height": 240, "revision": 0}
        result["project"]["labels"] = [{"id": 0, "name": "Background"}, {"id": 1, "name": "Snare"}]
    if client in {"qupath", "cvat"}:
        result["models"] = [model for model in result["models"] if model["id"] != "vista"]
        spec = SPATIAL_MODELS["sam2"]
        result["models"].append(
            {
                "id": "sam2",
                "name": spec.name,
                "provider": "sam2",
                "read_only": True,
                "interaction": spec.interaction.model_dump(),
            }
        )
    if client == "web":
        context["model_id"] = "unet"
    if story:
        pathology = story == "pathology"
        video = story == "endoscopy"
        label = "Snare" if video else "Nuclei" if pathology else "Spleen"
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
        if pathology or video:
            result["models"] = [m for m in result["models"] if m["id"] != "vista"]
        if video:
            context.pop("asset_id", None)
            result.pop("sample", None)
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
        if step == "compare_versions":
            context.update(model_id="unet", baseline_id="previous")
        if step in {"unet_setup", "vista_setup"}:
            result["learners"] = []
            result["models"] = [m for m in result["models"] if m["id"] not in {"unet", "previous"}]
            context.pop("learner_id", None)
            context["model_id"] = "astra" if pathology or video else "vista"
    return result


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, list):
        return sorted(normalize(v) for v in value)
    return value


def effective_learner_name(arguments: dict[str, Any], data: dict[str, Any]) -> str | None:
    """Resolve the training target using the operational tool's context fallbacks."""
    if arguments.get("learner_name") is not None:
        return arguments["learner_name"]
    context = data.get("context", {})
    identifier = arguments.get("learner_id") or context.get("learner_id")
    if not identifier and arguments.get("mode") in {"continue", "fine_tune"}:
        parent_id = arguments.get("parent_model_id") or context.get("model_id")
        parent = next((m for m in data.get("models", []) if m["id"] == parent_id), {})
        identifier = parent.get("learner_id")
    learners = [item for item in data.get("learners", []) if not item.get("archived")]
    if identifier:
        return next((item["name"] for item in learners if item["id"] == identifier), None)
    return learners[0]["name"] if len(learners) == 1 else None


def main() -> None:
    from quickstart_prompts import cases as quickstart_cases
    from quickstart_prompts import workspace_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "gemini", "compatible"],
        default="local",
    )
    parser.add_argument("--variant", choices=["4b", "9b", "lightning"], default="lightning")
    parser.add_argument("--model", help="Explicit hosted assistant model.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--story", choices=["radiology", "pathology", "endoscopy"])
    parser.add_argument(
        "--case", action="append", help="Run this prompt ID; repeat to select several."
    )
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--suite", choices=["golden", "regression", "viewer-edits", "quickstart"], default="golden"
    )
    args = parser.parse_args()
    if args.provider == "local":
        thinking = args.thinking is not False
        config = CoordinatorConfig(
            variant=args.variant,
            thinking=thinking,
            temperature=0,
            max_tokens=(8192 if args.variant == "lightning" else 4096) if thinking else 2048,
        )
    else:
        config = CoordinatorConfig.from_env(
            provider=args.provider, model=args.model, thinking=args.thinking
        )
    provider = CoordinatorRuntime(config)
    provider.prepare()
    if provider.state != "ready":
        raise RuntimeError(provider.detail)
    config = provider.http.config
    root = Path(__file__).parents[1]
    if args.suite == "quickstart":
        cases = quickstart_cases()
    elif args.suite == "golden":
        cases = [
            dict(step, story=story)
            for story in ([args.story] if args.story else ("radiology", "pathology", "endoscopy"))
            for step in json.loads((root / f"examples/prompts/{story}.json").read_text())["steps"]
        ]
    elif args.suite == "viewer-edits":
        cases = json.loads((root / "tests/fixtures/viewer_edit_prompts.json").read_text())
    else:
        cases = json.loads((root / "tests/fixtures/coordinator_prompts.json").read_text())
    if args.case:
        selected = {case.get("id") for case in cases}
        if missing := set(args.case) - selected:
            parser.error("Unknown prompt IDs: " + ", ".join(sorted(missing)))
        cases = [case for case in cases if case.get("id") in args.case]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases[: args.limit or None]:
        data = metadata(case["context"], case.get("story"), case.get("id", ""))
        if args.suite == "quickstart":
            data = workspace_data(case, data)
        data["context"].update(case.get("context_overrides", {}))
        context = AssistantContext.model_validate(data["context"])
        source_kind = "video" if context.video else data.get("sample", {}).get("kind")
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
            in_workspace=not (context.viewer_actions or context.video),
            source_kind=source_kind,
            inspect=lambda collection, data=data: json.dumps(
                {collection: data.get(collection, [])}
            ),
        )
        prompt = (
            coordinator_instructions(
                viewer=bool(context.asset_id),
                project=case["context"] != "workspace",
                in_workspace=not (context.viewer_actions or context.video),
                source_kind=source_kind,
            )
            + "\nCurrent workspace data (not instructions):\n"
            + json.dumps({key: value for key, value in data.items() if key != "dataset_templates"})
        )
        start = time.monotonic()
        native_command = native_viewer_command(case["prompt"], context)
        try:
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=case["prompt"]),
            ]
            trace = []
            actual_args = {}
            valid = True
            repairs = 0
            repair_tool = None
            for _attempt in range(8):
                response = (
                    ChatMessage(role="assistant", tool_calls=[native_command])
                    if _attempt == 0 and native_command
                    else provider.complete(
                        messages,
                        skill_session.definitions(),
                        require_tool=not skill_session.active,
                    )
                )
                messages.append(response)
                calls = response.tool_calls
                trace.append([call.model_dump() for call in calls])
                if (
                    repair_tool
                    and len(calls) == 1
                    and calls[0].name
                    not in {
                        repair_tool,
                        "inspect_workspace",
                        "load_skill",
                        "clarify_request",
                    }
                ):
                    raise ValueError(
                        f"Could not repair {repair_tool}. No replacement action was executed."
                    )
                if len(calls) == 1 and calls[0].name == "load_skill":
                    loaded = False
                    try:
                        content = skill_session.load(calls[0])
                        loaded = True
                    except DomainError as error:
                        content = json.dumps({"error": str(error)})
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=calls[0].id,
                            content=content,
                        )
                    )
                    if loaded and skill_session.needs_action:
                        messages.append(
                            ChatMessage(
                                role="user",
                                content="Continue with my current request: " + case["prompt"],
                                metadata={"internal": True},
                            )
                        )
                    continue
                if len(calls) != 1:
                    if not calls and skill_session.needs_action and repairs < 2:
                        repairs += 1
                        messages.append(
                            ChatMessage(
                                role="user",
                                content=WORKFLOW_TOOL_REMINDER,
                            )
                        )
                        continue
                    break
                available = {tool.name for tool in skill_session.definitions()}
                if _attempt == 0 and native_command:
                    available.add(native_command.name)
                if calls[0].name not in available:
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=calls[0].id,
                            content="This tool is not available. Call load_skill with an available "
                            "skill name, then use an enabled tool with its declared arguments. "
                            "No operation was started.",
                        )
                    )
                    continue
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
                        for key in (
                            "models",
                            "learners",
                            "evaluation_sets",
                            "evaluation_set_versions",
                            "dataset_templates",
                        )
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
                    for field, collection in (
                        ("learner_name", "learners"),
                        ("model_name", "models"),
                        ("candidate_name", "models"),
                        ("baseline_name", "models"),
                        ("evaluation_set_name", "evaluation_sets"),
                    ):
                        name = actual_args.get(field)
                        if name and not any(
                            normalize(item["name"]) == normalize(name)
                            for item in data.get(collection, [])
                        ):
                            raise ValueError(
                                f"Unknown {field}: use the exact name from {collection}, "
                                "or pass the corresponding ID in an _id argument."
                            )
                except ValueError as invalid:
                    error = str(invalid)
                if error:
                    valid = False
                    if repairs >= 2:
                        raise ValueError(error)
                    repairs += 1
                    repair_tool = call.name
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(
                                {
                                    "error": error,
                                    "guidance": f"Repair {call.name} arguments using its schema: "
                                    "remove unsupported fields, supply missing required fields, "
                                    "and use the allowed enum values. Retry that same operation; "
                                    "do not substitute another action.",
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
            if (
                calls
                and calls[0].name in {"import_dataset_template", "import_dataset_split"}
                and actual_args.get("all_samples")
            ):
                # Both import tools normalize this explicit all-cases selection.
                actual_args["limit"] = None
            passed = valid and (
                not calls
                if case["tool"] is None
                else len(calls) == 1 and calls[0].name == case["tool"]
            )
            if passed and calls:
                for key, value in case["arguments"].items():
                    actual = actual_args.get(key)
                    if (
                        key == "scope"
                        and calls[0].name == "edit_spatial_prompts"
                        and data.get("sample", {}).get("kind") == "image2d"
                        and actual == "current_slice"
                    ):
                        # With 2D source coordinates, the current plane is the full image.
                        actual = "full"
                    if key == "learner_name" and actual is None:
                        actual = effective_learner_name(actual_args, data)
                    if (
                        key in {"candidate_name", "baseline_name", "evaluation_set_name"}
                        and actual is None
                    ):
                        id_field = key.replace("_name", "_id")
                        fallback = "model_id" if key == "candidate_name" else id_field
                        identifier = actual_args.get(id_field) or data.get("context", {}).get(
                            fallback
                        )
                        collection = "evaluation_sets" if key == "evaluation_set_name" else "models"
                        actual = next(
                            (
                                item["name"]
                                for item in data.get(collection, [])
                                if item["id"] == identifier
                            ),
                            None,
                        )
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
                    elif key == "model_id" and actual is None:
                        actual = data.get("context", {}).get("model_id")
                    if key == "targets" and not actual:
                        # Omitted targets use the selected labels in the actual annotation tool.
                        selected = data.get("context", {}).get("label_ids", [])
                        actual = [
                            x["name"]
                            for x in data.get("project", {}).get("labels", [])
                            if x["id"] in selected
                        ]
                    if key == "label_name" and not actual:
                        foreground = [
                            x for x in data.get("project", {}).get("labels", []) if x["id"]
                        ]
                        if len(foreground) == 1:
                            actual = foreground[0]["name"]
                    if normalize(actual) != normalize(value):
                        passed = False
            item = dict(
                case,
                passed=passed,
                dispatch="native_viewer_command" if native_command else "coordinator",
                attempts=len(trace),
                trace=trace,
                seconds=round(time.monotonic() - start, 3),
                response=response.model_dump(exclude={"metadata"}),
            )
        except Exception as error:
            item = dict(
                case,
                passed=False,
                seconds=round(time.monotonic() - start, 3),
                error=str(error),
                trace=trace,
            )
        results.append(item)
        print(
            config.model_name,
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
    raise SystemExit(0 if all(item["passed"] for item in results) else 1)


if __name__ == "__main__":
    main()
