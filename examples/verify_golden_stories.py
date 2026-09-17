"""Execute the golden learning stories in isolated synthetic workspaces.

Default: deterministic coordinator outputs, fixture annotation, real U-Net training.
--coordinator lightning (or 4b/9b): use that running local model to interpret the same prompts.
Native viewer rendering and real VISTA3D are separate checks; no hosted model is called.
"""

import argparse
import gzip
import io
import json
import tempfile
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import Learner, ModelRecord, Snapshot
from monailabel.core.ports import Prediction
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.app import create_app
from monailabel.server.coordinator_runtime import CoordinatorRuntime

ROOT = Path(__file__).resolve().parents[1]


class FixtureCoordinator:
    """Test double only: verifies tool execution independently of language understanding."""

    def __init__(self):
        self.response = None

    def complete(self, messages, tools):
        if messages[-1].role == "tool":
            return ChatMessage(role="assistant", content="Requested records are available.")
        assert self.response is not None
        response, self.response = self.response, None
        return response


class FixtureAnnotation:
    """Deliberately imperfect image threshold; never uses the reference mask."""

    def predict(self, image, labels, prompt, model):
        target = next(label.id for label in labels if label.id)
        return Prediction(np.where(image.mean(axis=-1) > 0.62, target, 0).astype(np.uint8))


def synthetic_case(pathology, seed):
    rng = np.random.default_rng(seed)
    shape = (40, 48) if pathology else (24, 24, 24)
    grid = np.indices(shape)
    center = np.array(shape) / 2 + rng.uniform(-3, 3, len(shape))
    radius = 7 if pathology else 5
    distance = sum(((grid[i] - center[i]) / radius) ** 2 for i in range(len(shape)))
    mask = (distance < 1).astype(np.uint8)
    image = np.clip(mask * 0.55 + 0.15 + rng.normal(0, 0.08, shape), 0, 1).astype(np.float32)
    if pathology:
        rgb = np.stack([image, image * 0.85, image * 0.7], axis=-1)
        output = io.BytesIO()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(output, format="PNG")
        return output.getvalue(), mask, ".png"
    source = nib.Nifti1Image(image, np.diag([-1.5, 1.5, 1.5, 1.0]))
    return gzip.compress(source.to_bytes()), mask, ".nii.gz"


def run_story(name, coordinator="fixture"):
    definition = json.loads((ROOT / f"examples/prompts/{name}.json").read_text())
    steps = {s["id"]: s for s in definition["steps"]}
    pathology = name == "pathology"
    provider = (
        FixtureCoordinator()
        if coordinator == "fixture"
        else CoordinatorRuntime(CoordinatorConfig(variant=coordinator))
    )
    if isinstance(provider, CoordinatorRuntime):
        # Runs synchronously in this standalone verifier, reusing a healthy runtime.
        provider.prepare()
        if provider.state != "ready":
            raise RuntimeError(provider.detail)
    with (
        tempfile.TemporaryDirectory(prefix=f"monailabel-golden-{name}-") as directory,
        TestClient(create_app(Path(directory), chat_provider=provider)) as http,
    ):
        client = Client(http=http)
        client.post(
            "/api/auth/setup",
            {"username": "fixture-owner", "password": "synthetic-story-password"},
        )
        service = http.app.state.services
        service.presets.enabled = False
        service.models.providers["openai-chat-polygons"] = FixtureAnnotation()
        pid = None
        context = {}
        conversations = {}
        log = []
        aliases = {}

        def prompt(key, extra=None):
            step = steps[key]
            arguments = dict(step["arguments"])
            if arguments.get("model_id") in aliases:
                arguments["model_id"] = aliases[arguments["model_id"]]
            if isinstance(provider, FixtureCoordinator):
                provider.response = ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id=uuid4().hex, name=step["tool"], arguments=arguments)],
                )
            request_context = context | (extra or {})
            conversation_key = (pid, request_context.get("asset_id"), step["context"])
            reply = client.post(
                "/api/assistant",
                {
                    "project_id": pid,
                    "message": step["prompt"],
                    "context": request_context,
                    "conversation_id": conversations.get(conversation_key),
                },
            )
            conversations[conversation_key] = reply["conversation_id"]
            assert step["tool"] in reply["tools"], (key, reply)
            for field in ("learner_id", "model_id", "snapshot_id", "evaluation_id"):
                if field in reply["data"]:
                    context[field] = reply["data"][field]
            log.append({"id": key, "prompt": step["prompt"], "tools": reply["tools"]})
            if reply["job_id"]:
                result = client.wait(reply["job_id"], timeout=600)
                for field in ("model_id", "snapshot_id", "evaluation_id"):
                    if field in result:
                        context[field] = result[field]
                return result
            return reply["data"]

        pid = prompt("project")["project"]["id"]
        prefix = f"/api/projects/{pid}"
        prompt("import")
        cases = []
        for seed, split in enumerate(["train"] * 3 + ["validation"] * 2 + ["pool"]):
            content, truth, suffix = synthetic_case(pathology, seed)
            response = http.post(
                prefix + "/assets/upload",
                params={
                    "name": f"synthetic-{seed}{suffix}",
                    "group_id": f"independent-source-{seed}",
                    "split": split,
                },
                content=content,
            )
            response.raise_for_status()
            cases.append((response.json(), truth))
        prompt("key")
        prompt("models")
        model = client.post(
            prefix + "/models",
            {
                "name": "Sol · synthetic annotation fixture (not hosted inference)",
                "provider": "openai-chat-polygons",
                "label_ids": [0],
                "config": {
                    "url": "http://127.0.0.1:9/v1/chat/completions",
                    "model": "synthetic-only",
                },
            },
        )
        aliases["sol"] = model["id"]
        context["model_id"] = model["id"]
        viewer = "qupath" if pathology else "slicer"
        capabilities = ["submit", "review_annotation"]
        if pathology:
            capabilities += ["clear_segments", "classify_objects", "undo", "redo", "save_draft"]

        def annotate_and_review(asset, truth, model_id):
            context.update(
                asset_id=asset["id"],
                model_id=model_id,
                base_revision=asset["revision"],
                viewer_actions=capabilities,
            )
            if pathology:
                context["image_region"] = {"x": 4, "y": 4, "width": 24, "height": 24}
                context["image_tiling"] = {"tile_size": 256, "overlap": 32}
            else:
                context["slice"] = {"axis": 2, "index": 12, "window": [0, 1]}
            assert prompt("viewer")["viewer"] == viewer
            if pathology and model_id == model["id"]:
                crop = prompt("region")
                region = client.get("/api/proposals/" + crop["proposal_id"])
                assert region["image_region"]["width"] == 24
                prediction = prompt("whole")
            else:
                prediction = prompt("annotate_v1" if pathology else "handoff")
            proposal = client.get("/api/proposals/" + prediction["proposal_id"])
            assert proposal["base_revision"] == asset["revision"]
            assert proposal["image_region"] is None
            assert client.get("/api/assets/" + asset["id"])["revision"] == asset["revision"]
            assert prompt("submit")["operation"] == "submit"
            # Simulated complete manual correction: fixtures are explicit, never model labels.
            submitted = client.post(
                "/api/assets/" + asset["id"] + "/review",
                {
                    "base_revision": asset["revision"],
                    "mask": truth.tolist(),
                    "covered_labels": [0, 1],
                    "proposal_id": proposal["id"],
                },
            )
            context["base_revision"] = submitted["revision"]
            review = prompt("accept")
            assert review["verdict"] == "accepted"
            response = http.post(
                "/api/assets/" + asset["id"] + "/review-complete",
                content=truth.tobytes(),
                headers={
                    "X-MONAILABEL-REVIEW": json.dumps(
                        {
                            "base_revision": submitted["revision"],
                            "covered_labels": [0, 1],
                            "comment": "Synthetic fixture review; not clinical annotation.",
                        }
                    )
                },
            )
            response.raise_for_status()
            assert response.json()["revision"] == submitted["revision"]
            context.pop("asset_id")
            context.pop("base_revision")
            context.pop("slice", None)
            context.pop("image_region", None)
            context.pop("image_tiling", None)
            context["viewer_actions"] = []

        # Two training + two validation sources; hold a third training case for round two.
        for index in (0, 1, 3, 4):
            annotate_and_review(*cases[index], model["id"])
        assert prompt("queue")["review_queue"] == []
        prompt("snapshot")
        setup = prompt("unet_setup")
        learner = service.store.get(Learner, setup["learner_id"])
        # Bound the smoke-test workload; production defaults and user projects are untouched.
        config = learner.config | {
            "patch_size": 16,
            "channels": [4, 8, 16, 32],
            "epochs": 1,
            "steps_per_epoch": 3,
            "device": "cpu",
        }
        with service.store.transaction() as session:
            session.update(learner.model_copy(update={"config": config}))
        first = prompt("train")
        version1 = service.store.get(ModelRecord, first["model_id"])
        original_state = service.artifacts.read(version1.state_key)
        # A new source is annotated with version 1, explicitly corrected and reviewed.
        annotate_and_review(*cases[2], version1.id)
        context.update(model_id=version1.id, learner_id=learner.id)
        second = prompt("continue")
        version2 = service.store.get(ModelRecord, second["model_id"])
        assert version2.parent_id == version1.id and version2.mode == "continue"
        assert len(version1.training_groups) == 1 and len(version2.training_groups) == 2
        assert service.artifacts.read(version1.state_key) == original_state
        assert version1.state_key != version2.state_key
        snapshot = service.store.get(Snapshot, second["snapshot_id"])
        reference = client.post(
            prefix + "/evaluation-sets",
            {
                "name": "Independent comparison references",
                "percentage": 100,
                "auto_update": False,
                "asset_ids": [cases[i][0]["id"] for i in (3, 4)],
            },
        )
        context.update(
            model_id=version2.id,
            baseline_id=version1.id,
            evaluation_set_id=reference["id"],
            snapshot_id=None,
        )
        evaluation_job = prompt("compare_versions")
        evaluation = client.get("/api/evaluations/" + evaluation_job["evaluation_id"])
        assert set(evaluation["validation_assets"]) == {cases[i][0]["id"] for i in (3, 4)}
        assert not set(version2.training_groups) & {cases[i][0]["group_id"] for i in (3, 4)}
        assert all(s.label_source == "reviewed" for s in snapshot.samples)
        assert prompt("scores")["evaluations"][-1]["id"] == evaluation["id"]
        # Inference on untouched pool data must leave its revision unchanged.
        pool, _ = cases[5]
        result = client.wait(
            client.post("/api/assets/" + pool["id"] + "/annotate", {"model_id": version2.id})["id"]
        )
        assert client.get("/api/proposals/" + result["proposal_id"])["model_ids"] == [version2.id]
        assert client.get("/api/assets/" + pool["id"])["revision"] == 0
        project = client.get(prefix)
        assert all(m != version2.id for m in project["defaults"].values())
        return {
            "story": name,
            "synthetic_only": True,
            "coordinator": coordinator,
            "annotation_provider": "local fixture; no hosted Sol/Astra inference",
            "training_recipe": "monai-unet",
            "spatial_dims": learner.config["spatial_dims"],
            "channels": learner.config["in_channels"],
            "round_one_training_sources": len(version1.training_groups),
            "round_two_training_sources": len(version2.training_groups),
            "held_out_sources": 2,
            "previous_dice": evaluation["baseline"]["mean_dice"],
            "new_dice": evaluation["candidate"]["mean_dice"],
            "parent_checkpoint_unchanged": True,
            "defaults_unchanged": True,
            "native_viewer_rendering_tested": False,
            "hosted_annotation_calls": 0,
            "steps": log,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story", choices=["radiology", "pathology", "both"], default="both")
    parser.add_argument(
        "--coordinator", choices=["fixture", "lightning", "4b", "9b"], default="fixture"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["radiology", "pathology"] if args.story == "both" else [args.story]
    reports = []
    for name in names:
        print("Running", name, "synthetic golden story", flush=True)
        reports.append(run_story(name, args.coordinator))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in reports[-1].items() if k != "steps"}, indent=2), flush=True
        )


if __name__ == "__main__":
    main()
