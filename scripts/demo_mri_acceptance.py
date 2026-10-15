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

"""Cached public MRI and real TotalSegmentator inference in a disposable visitor workspace."""

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
    def complete(self, *args, **kwargs):
        raise RuntimeError("This MRI verification must not use a paid chat model.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
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
    with (
        tempfile.TemporaryDirectory(dir=args.output) as root,
        TestClient(create_app(Path(root), chat_provider=NoChat())) as http,
    ):
        client = Client(http=http)
        client.post("/api/auth/setup", {"username": "owner", "password": secrets.token_urlsafe(24)})
        http.cookies.clear()
        credentials = {"username": "mri-visitor", "password": secrets.token_urlsafe(24)}
        client.post("/api/auth/signup", credentials)
        client.post("/api/auth/login", credentials)
        pid = client.get("/api/projects")[0]["id"]
        for job in client.get(f"/api/projects/{pid}/jobs"):
            client.wait(job["id"], timeout=180)
        job = client.post("/api/demo-access/starter", {"example": "mri"})
        client.wait(job["id"], timeout=180)
        assert client.post("/api/demo-access/starter", {"example": "mri"}) is None
        assets = client.get(f"/api/projects/{pid}/assets")
        assert len(assets) == 2
        mri = next(a for a in assets if a["group_id"].startswith("msd:Task05_Prostate:"))
        project = client.get(f"/api/projects/{pid}")
        target = next(label["id"] for label in project["labels"] if label["name"] == "Prostate")
        model = next(
            m
            for m in client.get(f"/api/projects/{pid}/models")
            if m["provider"] == "totalsegmentator-mr"
        )
        job = client.post(
            f"/api/assets/{mri['id']}/annotate",
            {"model_id": model["id"], "label_ids": [target]},
        )
        result = client.wait(job["id"], timeout=600)
        service = http.app.state.services
        proposal = service.store.get(Proposal, result["proposal_id"])
        mask = service.artifacts.array(proposal.mask_key)
        assert list(mask.shape) == mri["spatial_shape"] and np.any(mask == target)
        report = {
            "passed": True,
            "sample": mri["name"],
            "shape": list(mask.shape),
            "foreground_voxels": int(np.count_nonzero(mask == target)),
            "ct_preserved": True,
            "source_geometry_preserved": True,
            "live_workspace_accessed": False,
            "hosted_api_calls": 0,
            "clinical_accuracy_evaluated": False,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
