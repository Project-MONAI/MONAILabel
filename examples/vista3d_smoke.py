"""GPU workflow check with synthetic labels, isolated from the user's workspace.

uv run python examples/vista3d_smoke.py
Downloads/reuses the pinned VISTA3D weights. This checks plumbing, not accuracy.
"""

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolDefinition
from monailabel.core.models import ModelRecord, Snapshot
from monailabel.monai.vista_runtime import load_checkpoint
from monailabel.monai.vista_weights import pretrained_weights
from monailabel.server.app import create_app


class NoChat:
    """This API-only check must never provision or call a conversation model."""

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        require_tool: bool = False,
    ) -> ChatMessage:
        raise AssertionError("The VISTA3D smoke check must not call a chat model.")


def main() -> None:
    weights = pretrained_weights()
    with weights.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with (
        tempfile.TemporaryDirectory(prefix="monailabel-vista-smoke-") as directory,
        TestClient(create_app(Path(directory), chat_provider=NoChat())) as http,
    ):
        client = Client(http=http)
        client.post(
            "/api/auth/setup",
            {"username": "synthetic-owner", "password": "synthetic-test-password"},
        )
        http.app.state.services.presets.enabled = True
        project = client.post(
            "/api/projects",
            {
                "name": "Synthetic VISTA3D workflow",
                "labels": [
                    {"id": 0, "name": "Background", "color": "#000000"},
                    {"id": 5, "name": "Spleen", "color": "#00ff00"},
                    {"id": 9, "name": "Liver", "color": "#ff0000"},
                    {"id": 12, "name": "Other", "color": "#0000ff"},
                ],
            },
        )
        prefix = f"/api/projects/{project['id']}"
        base = next(m for m in client.get(prefix + "/models") if m["preset"] == "vista3d")
        mask = np.zeros((32, 32, 32), np.uint8)
        mask[4:12, 5:13, 6:14] = 5
        mask[19:27, 18:26, 17:25] = 9
        mask[0, 0, 0] = 12
        assets = []
        for seed, split in enumerate(("train", "validation")):
            case_mask = np.roll(mask, seed, axis=0)
            image = np.where(case_mask == 5, 60, np.where(case_mask == 9, 100, -900))
            image = (image + np.random.default_rng(seed).normal(0, 2, image.shape)).astype(
                np.float32
            )
            source = nib.Nifti1Image(image, np.diag([-1.5, 1.5, 1.5, 1.0]))
            response = http.post(
                prefix + "/assets/upload",
                params={"name": split + ".nii.gz", "group_id": split, "split": split},
                content=gzip.compress(source.to_bytes()),
            )
            response.raise_for_status()
            asset = response.json()
            assets.append(asset)
            annotation = client.post(
                f"/api/assets/{asset['id']}/review",
                {"base_revision": 0, "mask": case_mask.tolist(), "covered_labels": [0, 5, 9, 12]},
            )
            client.post(
                f"/api/annotations/{annotation['id']}/decision",
                {"verdict": "accepted", "comment": "Synthetic workflow fixture; not medical data."},
            )
        learner = client.post(
            prefix + "/learners",
            {
                "name": "Synthetic VISTA3D child",
                "recipe": "vista3d",
                "label_ids": [0, 5, 9],
                "initial_model_id": base["id"],
                "config": {"patch_size": 32, "epochs": 1, "steps_per_epoch": 1},
            },
        )
        job = client.post(
            prefix + f"/learners/{learner['id']}/train",
            {"mode": "fine_tune", "parent_model_id": base["id"]},
        )
        result = client.wait(job["id"], timeout=600)
        service = http.app.state.services
        child = service.store.get(ModelRecord, result["model_id"])
        assert child.parent_id == base["id"] and not child.read_only
        assert child.training_groups == ["train"]
        state = service.artifacts.json(child.state_key)
        assert state["steps"] == 1 and np.isfinite(state["final_loss"])
        checkpoint = load_checkpoint(state, service.artifacts)
        assert checkpoint["optimizer"]["state"]
        del checkpoint
        snapshot = service.store.get(Snapshot, result["snapshot_id"])
        assert [label.id for label in snapshot.labels] == [0, 5, 9]
        assert all(
            set(np.unique(service.artifacts.array(s.mask_key))) == {0, 5, 9}
            for s in snapshot.samples
        )
        evaluation_job = client.post(
            prefix + "/evaluate",
            {
                "snapshot_id": snapshot.id,
                "candidate_id": child.id,
                "baseline_id": base["id"],
                "label_ids": [5, 9],
            },
        )
        evaluation = client.wait(evaluation_job["id"], timeout=600)
        continuation = client.post(
            prefix + f"/learners/{learner['id']}/train",
            {"mode": "continue", "parent_model_id": child.id},
        )
        continued_result = client.wait(continuation["id"], timeout=600)
        continued = service.store.get(ModelRecord, continued_result["model_id"])
        continued_state = service.artifacts.json(continued.state_key)
        assert continued_state["steps"] == 2 and continued.parent_id == child.id
        assert service.store.get(ModelRecord, base["id"]).model_dump(mode="json") == base
        with weights.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == checksum
        assert not weights.stat().st_mode & 0o222
        print(
            json.dumps(
                {
                    "synthetic_only": True,
                    "base_unchanged": True,
                    "filtered_labels": [0, 5, 9],
                    "vista_mapping": learner["config"]["label_mapping"],
                    "training_cases": 1,
                    "held_out_cases": 1,
                    "fine_tune_steps": state["steps"],
                    "continued_steps": continued_state["steps"],
                    "initial_loss": state["initial_loss"],
                    "final_loss": continued_state["final_loss"],
                    "held_out_evaluation_completed": bool(evaluation["evaluation_id"]),
                    "hosted_api_calls": 0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
