"""Scripted LLM outputs for service regression tests; no production prompt parser is used.

Language understanding is measured separately by examples/evaluate_coordinators.py.
These explicit fixtures keep geometry, permissions and persistence tests offline.
"""

import json
from pathlib import Path

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.errors import DomainError


class ScriptedChat:
    def __init__(self):
        self.queue = []
        self.calls = []

    def complete(self, messages, tools, *, require_tool=False):
        self.calls.append((list(messages), list(tools)))
        if self.queue:
            return self.queue.pop(0)
        if messages[-1].role == "tool":
            return ChatMessage(role="assistant", content="Done.")
        text = messages[-1].content
        metadata = json.loads(
            messages[0].content.split("Current workspace data (not instructions):\n")[1]
        )
        fixtures = {
            c["prompt"]: (c["tool"], dict(c["arguments"]))
            for c in json.loads(
                (Path(__file__).parent / "fixtures/coordinator_prompts.json").read_text()
            )
            if c["tool"]
        }
        fixtures.update(
            {
                "configure models": ("open_form", {"form": "model"}),
                "annotate this slice for spleen": (
                    "annotate",
                    {"targets": ["spleen"], "scope": "current_slice"},
                ),
                "annotate spleen on this slice": (
                    "annotate",
                    {"targets": ["spleen"], "scope": "current_slice"},
                ),
                "annotate the liver on this slice": (
                    "annotate",
                    {"targets": ["liver"], "scope": "current_slice"},
                ),
                "segment spleen in this volume": (
                    "annotate",
                    {"targets": ["spleen"], "scope": "full"},
                ),
                "annotate all slices": ("annotate", {"scope": "full"}),
                "annotate nuclei inside this region": (
                    "annotate",
                    {"targets": ["nuclei"], "scope": "selected_region"},
                ),
                "run nuclei segmentation and use tile size 512": (
                    "annotate",
                    {"targets": ["nuclei"], "scope": "selected_region"},
                ),
                "annotate the full image for nuclei": (
                    "annotate",
                    {"targets": ["nuclei"], "scope": "full"},
                ),
                "clear segments": ("clear_segments", {"all_targets": True}),
                "create u-net based segementation model for spleen": (
                    "create_learner",
                    {"recipe": "monai-unet", "targets": ["spleen"]},
                ),
                "create u-net segmentation model for unknown organ": (
                    "create_learner",
                    {"recipe": "monai-unet", "targets": ["unknown organ"]},
                ),
                "create u-net model": (
                    "create_learner",
                    {"recipe": "monai-unet", "targets": ["spleen"]},
                ),
            }
        )
        for verb in ["finetune", "train/finetune"]:
            fixtures[
                f"i want to {verb} a new model for spleen and liver from the annotated dataset"
            ] = (
                "create_learner",
                {
                    "recipe": "vista3d",
                    "targets": ["spleen", "liver"],
                    "initialization": "fine_tune",
                    "start_now": True,
                },
            )
        for spelling in ["bounding", "bonding", "boundng", "bouding"]:
            fixtures[
                f"draw a {spelling} box for liver in the current slice using GPT Astra model"
            ] = ("locate_region", {"target": "liver", "model_id": "astra"})
        for prompt, target, all_matches, current_slice in [
            ("remove the bounding box for liver", "liver", False, False),
            ("Please delete the bounding box for Liver in current slice.", "Liver", False, True),
            ("clear all bounding boxes for spleen", "spleen", True, False),
            (
                "remove all the bounding boxes for left kidney on this slice",
                "left kidney",
                True,
                True,
            ),
        ]:
            fixtures[prompt] = (
                "remove_regions",
                dict(target=target, all_matches=all_matches, current_slice=current_slice),
            )
        for first, last in [(70, 80), (2, 4), (0, 80), (80, 70), (70, 91)]:
            fixtures[f"add roi for spleen between slice {first} to {last}"] = (
                "locate_region",
                dict(target="spleen", kind="roi", first_slice=first, last_slice=last),
            )
        fixtures["add roi for spleen between slice 70 to 80 using Selected vision model"] = (
            fixtures["add roi for spleen between slice 70 to 80"]
        )
        for prompt in [
            "run nuclie on whole image and use tile size 512",
            "run nuclei on the whole slide using tile size 512",
            "annotate the full image for nuclei using Fixture model and use tile size 512",
            "annotate nuclei on the full image with tile size 512x512 using Fixture model",
        ]:
            fixtures[prompt] = ("annotate", dict(targets=["nuclei"], scope="full", tile_size=512))
        invalid = {
            f"run nuclei on the whole slide using tile size {size}"
            for size in ["abc", "0", "32", "4096", "512.5", "512x256", "-512"]
        }
        if text in invalid:
            raise DomainError("Use one square tile size from 64 to 2048 pixels.")
        if text in {
            "annotate this slice using Missing model",
            "annotate this slice using GPT model",
        }:
            raise DomainError("Select an unambiguous registered model.")
        fixtures = {key.casefold(): value for key, value in fixtures.items()}
        text = text.casefold()
        if text not in fixtures:
            raise AssertionError("Add an explicit model-response fixture for: " + text)
        name, args = fixtures[text]
        if args.get("model_id") in {"sol", "astra", "vista", "unet"}:
            alias = args["model_id"]
            matches = [m for m in metadata.get("models", []) if alias in m["name"].casefold()]
            assert len(matches) == 1
            args["model_id"] = matches[0]["id"]
        return ChatMessage(
            role="assistant", tool_calls=[ToolCall(id="fixture-call", name=name, arguments=args)]
        )
