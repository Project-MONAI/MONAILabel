"""Run the README's eight prompts in a disposable workspace.

Fixture mode tests contracts without downloads or a GPU. --real uses Decathlon
and VISTA3D with recommended settings. --coordinator-url tests language understanding
against an existing local endpoint; no hosted service is started by this script.
"""

import argparse
import gzip
import io
import json
import tarfile
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, Conversation, ToolCall
from monailabel.core.models import Annotation, ModelRecord, Snapshot
from monailabel.core.ports import Prediction
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.http import HttpChat
from monailabel.server.app import create_app
from monailabel.server.coordinator_runtime import CoordinatorRuntime
from monailabel.server.dataset_downloads import Downloads
from monailabel.server.instructions import instruction_revision, skills

DEFINITION = Path(__file__).parent / "prompts/spleen.json"


class RecordingCoordinator:
    """Record tool routing, not private reasoning, for disposable workflow diagnostics."""

    def __init__(self, provider):
        self.provider = provider
        self.trace = []

    def complete(self, messages, tools, *, require_tool=False):
        item = {"available_tools": [tool.name for tool in tools]}
        self.trace.append(item)
        response = self.provider.complete(messages, tools, require_tool=require_tool)
        item["calls"] = [call.model_dump(exclude={"id"}) for call in response.tool_calls]
        return response


class FixtureCoordinator:
    """Explicit tool-call fixture: does not claim to test prompt interpretation."""

    response = None

    def complete(self, messages, tools, *, require_tool=False):
        assert self.response is not None
        if self.response.tool_calls[0].name not in {tool.name for tool in tools}:
            name = next(
                s.header.name for s in skills() if self.response.tool_calls[0].name in s.tools
            )
            return ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id=uuid4().hex, name="load_skill", arguments={"name": name})],
            )
        response, self.response = self.response, None
        return response


class FixtureVista:
    """Image-only prediction and deterministic checkpoint fixture, never real VISTA3D."""

    def predict_volume(self, volume, labels, prompt, model):
        threshold = 0.65 if model.read_only else 0.55
        return Prediction((volume.image[..., 0] > threshold).astype(np.uint8))

    def train_volumes(self, samples, label_ids, mode, parent_state, progress):
        assert parent_state == {"format": "vista3d-base-v1"}
        assert len(list(samples)) == 5 and label_ids == [0, 1]
        progress(1)
        return {"format": "fixture-only", "initial_loss": 1.0, "final_loss": 0.5}


def fixture_archive(path):
    """41 independent tiny volumes exercise the exact 32/9 import rounding."""
    rng = np.random.default_rng(7)
    with tarfile.open(path, "w") as archive:

        def add(name, data):
            item = tarfile.TarInfo("Task09_Spleen/" + name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))

        add("dataset.json", json.dumps({"labels": {"0": "background", "1": "spleen"}}).encode())
        for index in range(41):
            mask = np.zeros((12, 12, 12), np.uint8)
            mask[3:8, 3:8, 3:8] = 1
            image = (mask * 0.5 + rng.uniform(0, 0.3, mask.shape)).astype(np.float32)
            for folder, values in (("imagesTr", image), ("labelsTr", mask)):
                add(
                    f"{folder}/spleen_{index:02}.nii.gz",
                    gzip.compress(nib.Nifti1Image(values, np.eye(4)).to_bytes()),
                )


def run_workflow(
    *,
    real=False,
    coordinator="fixture",
    coordinator_url=None,
    coordinator_model=None,
    cache_dir=None,
    output=None,
    workspace=None,
    prompts=DEFINITION,
):
    definition = json.loads(prompts.read_text())
    provider = (
        HttpChat(
            CoordinatorConfig(
                provider="compatible",
                base_url=coordinator_url,
                model=coordinator_model,
                timeout=180,
            )
        )
        if coordinator_url
        else FixtureCoordinator()
    )
    if coordinator != "fixture":
        if coordinator_url:
            raise ValueError("Choose a managed coordinator or an existing endpoint.")
        provider = CoordinatorRuntime(CoordinatorConfig(variant=coordinator, timeout=180))
        provider.prepare()
        if provider.state != "ready":
            raise RuntimeError(provider.detail)
    report = {
        "real_vista3d": real,
        "real_coordinator": bool(coordinator_url) or coordinator != "fixture",
        "coordinator": coordinator_model or coordinator,
        "instruction_revision": instruction_revision(),
        "steps": [],
    }

    def save():
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="monailabel-spleen-golden-") as temporary:
        directory = workspace or Path(temporary) / "workspace"
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Verification requires an empty workspace, never the live workspace.")
        with TestClient(create_app(directory, chat_provider=provider)) as http:
            client = Client(http=http)
            client.post(
                "/api/auth/setup", {"username": "demo", "password": "disposable-demo-password"}
            )
            services = http.app.state.services
            recording = RecordingCoordinator(provider)
            services.assistants.provider = recording
            services.presets.enabled = True
            if real:
                if cache_dir:
                    services.dataset_templates.downloads = Downloads(cache_dir)
            else:
                archive = Path(temporary) / "fixture.tar"
                fixture_archive(archive)
                services.dataset_templates.downloads.fetch = lambda *_: archive
                services.models.recipes.segmenter = lambda *_: FixtureVista()
                services.models.recipes.trainer = lambda *_: FixtureVista()
            project_id = conversation_id = None
            context = {}
            evaluation_set_id = None
            imported_ids = annotation_ids = evaluation_ids = []
            base = trained = snapshot = None
            defaults = None
            for step in definition["steps"]:
                recording.trace = []
                reply = None
                try:
                    started = time.monotonic()
                    print(f"{step['id']}: {step['prompt']}", flush=True)
                    arguments = {
                        k: evaluation_set_id if v == "$evaluation_set_id" else v
                        for k, v in step["arguments"].items()
                    }
                    if isinstance(provider, FixtureCoordinator):
                        provider.response = ChatMessage(
                            role="assistant",
                            tool_calls=[
                                ToolCall(id=uuid4().hex, name=step["tool"], arguments=arguments)
                            ],
                        )
                    reply = client.post(
                        "/api/assistant",
                        {
                            "project_id": project_id,
                            "conversation_id": conversation_id,
                            "message": step["prompt"],
                            "context": context,
                        },
                    )
                    assert step["tool"] in reply["tools"], (step["id"], reply)
                    conversation_id = reply["conversation_id"]
                    for key in ("learner_id", "model_id", "snapshot_id"):
                        if key in reply["data"]:
                            context[key] = reply["data"][key]
                    result = reply["data"]
                    if reply["job_id"]:
                        result = client.wait(reply["job_id"], timeout=7200)
                        assert not result.get("failed"), result
                        for key in ("model_id", "snapshot_id"):
                            if key in result:
                                context[key] = result[key]
                    if step["id"] == "project":
                        project_id = result["project"]["id"]
                        conversation_id = None
                        prefix = f"/api/projects/{project_id}"
                        base = next(
                            m for m in client.get(prefix + "/models") if m["preset"] == "vista3d"
                        )
                    elif step["id"] == "import":
                        imported_ids = result["asset_ids"]
                        annotation_ids = result["annotation_asset_ids"]
                        evaluation_ids = result["evaluation_asset_ids"]
                        evaluation_set_id = result["evaluation_set_id"]
                        assert (
                            len(imported_ids) == 41
                            and len(annotation_ids) == 32
                            and len(evaluation_ids) == 9
                        )
                        assert not set(annotation_ids) & set(evaluation_ids)
                        assets = client.get(prefix + "/assets")
                        assert all(
                            bool(a["annotation_id"]) == (a["id"] in evaluation_ids) for a in assets
                        )
                    elif step["id"] == "review_evaluation":
                        decisions = client.get(prefix + "/decisions")
                        assert len(decisions) == 9 and {d["asset_id"] for d in decisions} == set(
                            evaluation_ids
                        )
                        assert all(d["verdict"] == "accepted" for d in decisions)
                    elif step["id"] == "annotate":
                        assert result["asset_ids"] == annotation_ids[:5]
                        assert len(result["annotation_ids"]) == 5
                        assert len(client.get(prefix + "/decisions")) == 9
                        assert all(
                            services.store.get(Annotation, i).proposal_id
                            for i in result["annotation_ids"]
                        )
                        assert client.get(f"/api/jobs/{reply['job_id']}/logs")["total_lines"] > 5
                    elif step["id"] == "review_training":
                        decisions = client.get(prefix + "/decisions")
                        assert len(decisions) == 14 and all(
                            d["verdict"] == "accepted" for d in decisions
                        )
                    elif step["id"] == "model":
                        learners = client.get(prefix + "/learners")
                        assert (
                            len(learners) == 1 and learners[0]["name"] == step["arguments"]["name"]
                        )
                        assert learners[0]["initial_model_id"] == base["id"]
                        assert not reply["job_id"]
                        defaults = client.get(prefix)["defaults"]
                    elif step["id"] == "train":
                        trained = services.store.get(ModelRecord, result["model_id"])
                        snapshot = services.store.get(Snapshot, result["snapshot_id"])
                        assert trained.parent_id == base["id"] and trained.mode == "fine_tune"
                        assert set(trained.training_assets) == set(annotation_ids[:5])
                        assert snapshot.evaluation_version_id
                        metrics = client.get(f"/api/jobs/{reply['job_id']}/training-report")
                        assert metrics["metrics"] and not metrics["error"], metrics
                        report["training_metrics"] = metrics["metrics"]
                    elif step["id"] == "compare":
                        evaluation = client.get("/api/evaluations/" + result["evaluation_id"])
                        assert (
                            evaluation["candidate_id"] == trained.id
                            and evaluation["baseline_id"] == base["id"]
                        )
                        assert evaluation["evaluation_version_id"] == snapshot.evaluation_version_id
                        assert set(evaluation["validation_assets"]) == set(evaluation_ids)
                        assert not set(evaluation_ids) & set(trained.training_assets)
                        assert (
                            services.store.get(ModelRecord, base["id"]).model_dump(mode="json")
                            == base
                        )
                        assert client.get(prefix)["defaults"] == defaults
                        report["comparison"] = evaluation
                    report["project_id"] = project_id
                    report["steps"].append(
                        {
                            "id": step["id"],
                            "prompt": step["prompt"],
                            "tools": reply["tools"],
                            "skills": [
                                call.arguments.get("name")
                                for message in services.store.get(
                                    Conversation, reply["conversation_id"]
                                )
                                .turns[-1]
                                .messages
                                for call in message.tool_calls
                                if call.name == "load_skill"
                            ],
                            "job_id": reply["job_id"],
                            "seconds": round(time.monotonic() - started, 2),
                            "trace": recording.trace,
                        }
                    )
                    save()
                    print(f"PASS {step['id']}", flush=True)
                except Exception as error:
                    report["failure"] = {
                        "step": step["id"],
                        "prompt": step["prompt"],
                        "error": str(error) or type(error).__name__,
                        "reply": reply,
                        "trace": recording.trace,
                    }
                    save()
                    raise
            report["counts"] = {
                "annotation_images": 32,
                "evaluation_images": 9,
                "trained_images": 5,
            }
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--coordinator", choices=("fixture", "4b", "9b", "lightning"), default="fixture"
    )
    parser.add_argument("--coordinator-url")
    parser.add_argument("--coordinator-model")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--workspace", type=Path, help="Optional empty workspace to retain results."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompts", type=Path, default=DEFINITION, help="Workflow prompt definition."
    )
    args = parser.parse_args()
    run_workflow(
        real=args.real,
        coordinator=args.coordinator,
        coordinator_url=args.coordinator_url,
        coordinator_model=args.coordinator_model,
        cache_dir=args.cache_dir,
        output=args.output,
        workspace=args.workspace,
        prompts=args.prompts,
    )


if __name__ == "__main__":
    main()
