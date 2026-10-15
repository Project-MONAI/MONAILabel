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

"""Exercise prepared examples and real local inference as a restricted visitor.

Uses a newly created disposable workspace, cached public data and local models.
No API key, live workspace, downloads or application service restart is used.
Geometric prompts below test execution, not anatomical accuracy.
"""

import argparse
import json
import os
import secrets
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.core.models import Proposal
from monailabel.server.app import create_app


class NoChat:
    def complete(self, messages, tools, *, require_tool=False):
        raise RuntimeError("This local-model test must never call a chat provider.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    for flag in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        os.environ["MONAILABEL_DEMO_" + flag] = "1"
    os.environ["MONAILABEL_PREPARED_MODELS_ONLY"] = "1"
    os.environ["MONAILABEL_PRELOAD_MODELS"] = "0"
    report = {
        "live_workspace_accessed": False,
        "hosted_api_calls": 0,
        "clinical_accuracy_evaluated": False,
        "steps": [],
    }
    args.output.mkdir(mode=0o700, exist_ok=False)
    with (
        tempfile.TemporaryDirectory(prefix="workspace-", dir=args.output) as temp,
        TestClient(create_app(Path(temp), chat_provider=NoChat())) as http,
    ):
        client = Client(http=http)
        client.post(
            "/api/auth/setup",
            {"username": "stage-owner", "password": secrets.token_urlsafe(24)},
        )
        foreign = client.post("/api/projects", {"name": "Private host project"})
        http.cookies.clear()
        credentials = {"username": "stage-visitor", "password": secrets.token_urlsafe(24)}
        client.post("/api/auth/signup", credentials)
        client.post("/api/auth/login", credentials)
        project = client.get("/api/projects")[0]
        pid = project["id"]
        for job in client.get(f"/api/projects/{pid}/jobs"):
            client.wait(job["id"], timeout=120)
        for choice in ("pathology", "endoscopy"):
            job = client.post("/api/demo-access/starter", {"example": choice})
            client.wait(job["id"], timeout=180)
            assert client.post("/api/demo-access/starter", {"example": choice}) is None
            report["steps"].append({"example": choice, "cached_import": "passed"})
        assert http.get(f"/api/projects/{foreign['id']}").status_code == 403
        assert http.post("/api/projects", json={"name": "Forbidden"}).status_code == 403
        assets = client.get(f"/api/projects/{pid}/assets")
        videos = client.get(f"/api/projects/{pid}/videos")
        assert len(assets) == 2 and len(videos) == 1
        models = client.get(f"/api/projects/{pid}/models")
        models = {model["provider"]: model for model in models}
        assert {"sam2", "nninteractive"} <= models.keys()
        project = client.get(f"/api/projects/{pid}")
        labels = {label["name"]: label["id"] for label in project["labels"]}
        assert {"Spleen", "Nucleus", "Snare"} <= labels.keys()
        for asset in assets:
            shape = asset["spatial_shape"]
            volume = asset["kind"] == "volume3d"
            provider = "nninteractive" if volume else "sam2"
            label = labels["Spleen" if volume else "Nucleus"]
            center = [size // 2 for size in shape]
            request = {
                "model_id": models[provider]["id"],
                "label_ids": [label],
                "spatial_prompt": {"points": [{"coordinates": center}]},
            }
            if volume:
                request.update(
                    all_slices=True,
                    slice={"axis": 2, "index": center[2], "window": [-1000, 1000]},
                )
            else:
                request["image_region"] = {
                    "x": center[1] - 64,
                    "y": center[0] - 64,
                    "width": 128,
                    "height": 128,
                }
            job = client.post(f"/api/assets/{asset['id']}/annotate", request)
            result = client.wait(job["id"], timeout=180)
            services = http.app.state.services
            proposal = services.store.get(Proposal, result["proposal_id"])
            mask = services.artifacts.array(proposal.mask_key)
            assert list(mask.shape) == shape and np.any(mask == label)
            client.post(
                f"/api/assets/{asset['id']}/review",
                {"base_revision": 0, "proposal_id": proposal.id, "covered_labels": [label]},
            )
            report["steps"].append(
                {
                    "provider": provider,
                    "sample": asset["name"],
                    "source_geometry_preserved": True,
                    "foreground_voxels": int(np.count_nonzero(mask)),
                    "submitted_for_review": True,
                }
            )
        report.update(passed=True, private_samples=3, role="annotator+reviewer")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
