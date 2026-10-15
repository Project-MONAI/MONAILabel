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

"""Prepare genuine public-data training examples in a NEW, private offline workspace.

Requires already cached Decathlon Spleen and pinned VISTA3D weights. No chat,
hosted models, downloads, live-workspace access, or fake training records.
The output retains real checkpoints and logs; publish only its catalog directory.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.core.models import ModelRecord, Snapshot, TrainingReport
from monailabel.providers.vista3d import REVISION
from monailabel.server.app import create_app
from monailabel.server.dataset_downloads import Downloads, sources
from monailabel.server.demo_training import RecordedTraining


class NoChat:
    def complete(self, *args, **kwargs):
        raise RuntimeError("Offline training preparation must never call a chat model.")


def wait(client, job):
    try:
        return client.wait(job["id"], timeout=3600)
    except BaseException:
        client.post(f"/api/jobs/{job['id']}/cancel")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new, nonexistent directory.")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--nnunet-epochs", type=int, default=20, choices=range(1, 101))
    parser.add_argument("--steps", type=int, default=32, choices=range(1, 101))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--source-revision must be the exact 40-character source commit.")
    # Explicit operator preparation, not a bypass available through the server API.
    os.environ["MONAILABEL_DEMO_TRAINING_PREVIEW"] = "0"
    os.environ["MONAILABEL_DEMO_CACHE_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MONAILABEL_PRELOAD_MODELS"] = "0"
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "NV_INFERENCE_API_KEY"):
        os.environ.pop(key, None)
    cache = Downloads(Path(os.environ["MONAILABEL_DATASETS_DIR"]))
    source = next(item for item in sources() if item.id == "Task09_Spleen")
    if not cache._cached_here(source):
        raise SystemExit("Prepare and verify the Spleen cache before offline training.")
    weights = Path(os.environ["MONAILABEL_MODELS_DIR"]) / "vista3d" / REVISION / "model.pt"
    if not weights.is_file():
        raise SystemExit("Prepare the pinned VISTA3D weights before offline training.")
    # Exclusive creation prevents accidentally opening an existing/live workspace.
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    catalog = args.output / "catalog"
    catalog.mkdir(mode=0o700)
    with TestClient(create_app(args.output / "workspace", chat_provider=NoChat())) as http:
        client = Client(http=http)
        client.post(
            "/api/auth/setup",
            {"username": "offline-preparation", "password": secrets.token_urlsafe(32)},
        )
        service = http.app.state.services
        service.presets.enabled = True
        project = client.post(
            "/api/projects",
            {
                "name": "Prepared Spleen training examples",
                "labels": [
                    {"id": 0, "name": "Background"},
                    {"id": 1, "name": "Spleen"},
                ],
            },
        )
        prefix = f"/api/projects/{project['id']}"
        for split, offset, count in (("train", 0, 3), ("validation", 3, 1)):
            print(f"Importing {count} cached public cases for {split}.", flush=True)
            wait(
                client,
                client.post(
                    prefix + "/dataset-imports",
                    {
                        "template_id": source.id,
                        "include_masks": True,
                        "split": split,
                        "offset": offset,
                        "limit": count,
                    },
                ),
            )
        assets = client.get(prefix + "/assets")
        # Dataset-provided references are fixtures, not claimed clinician corrections.
        for asset in assets:
            if asset["split"] == "train":
                client.post(
                    f"/api/annotations/{asset['annotation_id']}/decision",
                    {
                        "verdict": "accepted",
                        "comment": "Public dataset reference used for an offline workflow example; "
                        "not a clinician-reviewed demo annotation.",
                    },
                )
        base = next(m for m in client.get(prefix + "/models") if m["preset"] == "vista3d")
        for recipe in ("nnunet-v2", "vista3d"):
            config = (
                {"modality": "CT", "epochs": args.nnunet_epochs, "steps_per_epoch": args.steps}
                if recipe == "nnunet-v2"
                else {"epochs": 5, "steps_per_epoch": 10, "patch_size": 128}
            )
            learner = client.post(
                prefix + "/learners",
                {
                    "name": "Prepared " + recipe + " spleen example",
                    "recipe": recipe,
                    "label_ids": [0, 1],
                    "config": config,
                    "initial_model_id": base["id"] if recipe == "vista3d" else None,
                },
            )
            print(f"Starting genuine {recipe} run: {config}.", flush=True)
            started = time.monotonic()
            job = client.post(
                prefix + f"/learners/{learner['id']}/train",
                {
                    "mode": "fine_tune" if recipe == "vista3d" else "scratch",
                    "parent_model_id": base["id"] if recipe == "vista3d" else None,
                },
            )
            result = wait(client, job)
            elapsed = time.monotonic() - started
            model = service.store.get(ModelRecord, result["model_id"])
            snapshot = service.store.get(Snapshot, result["snapshot_id"])
            report = service.store.get(TrainingReport, result["training_report_id"])
            state = service.artifacts.json(model.state_key)
            checkpoint = service.artifacts.path(state["checkpoint_key"])
            with checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            losses = state.get("training", {}).get("losses", [])
            recorded = RecordedTraining(
                recipe=recipe,
                name=learner["name"],
                dataset="Medical Decathlon · Spleen (CT)",
                completed_at=datetime.now(UTC),
                duration_seconds=elapsed,
                training_cases=len(model.training_assets),
                evaluation_cases=len(report.validation_assets),
                epochs=config["epochs"],
                steps_per_epoch=config["steps_per_epoch"],
                initial_loss=report.initial_loss
                if report.initial_loss is not None
                else (losses[0] if losses else None),
                final_loss=report.final_loss
                if report.final_loss is not None
                else (losses[-1] if losses else None),
                mean_dice=report.metrics.mean_dice if report.metrics else None,
                checkpoint_sha256=checksum,
                source_revision=args.source_revision,
                caveats=[
                    "Real limited-budget training, not evidence of clinical accuracy "
                    "or improvement.",
                    "Public dataset reference masks; not corrections made by demo visitors.",
                    "Evaluation patients are excluded from this run's training. "
                    "Upstream base-model "
                    "pretraining overlap is not ruled out; this is not an independent benchmark.",
                    "Duration includes training preparation, checkpoint saving "
                    "and requested evaluation.",
                ]
                + (["Evaluation was unavailable: " + report.error] if report.error else []),
                log="".join(service.jobs.download_log(job["id"])),
            )
            assert len(model.training_assets) == 3
            assert not set(model.training_assets) & set(report.validation_assets)
            assert snapshot.project_id == project["id"]
            with (catalog / (recipe + ".json")).open("x") as output:
                output.write(recorded.model_dump_json(indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "recipe": recipe,
                        "seconds": round(elapsed, 2),
                        "mean_dice": recorded.mean_dice,
                        "recorded_example_saved": True,
                    }
                ),
                flush=True,
            )
    print(
        "Offline examples complete. Only the catalog directory is intended for visitor display.",
        flush=True,
    )


if __name__ == "__main__":
    main()
