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

"""Exercise installed distributions without a checkout, model download or external service."""

import gc
import importlib
import os
import secrets
import tempfile
from importlib.metadata import distribution, distributions, version
from importlib.resources import files
from pathlib import Path


def main():
    expected = version("monailabel")
    installed = {
        p.metadata["Name"] for p in distributions() if p.metadata["Name"].startswith("monailabel")
    }
    assert installed == {"monailabel"}, installed
    package = distribution("monailabel")
    direct = package.read_text("direct_url.json") or ""
    assert '"editable": true' not in direct and '"vcs_info"' not in direct
    for name in (
        "core",
        "client",
        "server",
        "providers",
        "monai",
        "sam",
        "dicom",
        "viewers",
        "totalsegmentator",
        "nninteractive",
    ):
        module = importlib.import_module("monailabel." + name)
        assert "site-packages" in Path(module.__file__).parts, module.__file__
    from monailabel.launcher import CLIENT_COMMANDS

    from monailabel.totalsegmentator.catalog import targets
    from monailabel.totalsegmentator.runtime import TotalSegmenter, TotalTrainer

    assert len(targets("totalsegmentator-ct")) == 117
    assert len(targets("totalsegmentator-mr")) == 50
    assert TotalSegmenter and TotalTrainer and "login" in CLIENT_COMMANDS
    import monailabel

    assert monailabel.__file__ is None, "The shared namespace must not have an __init__.py"
    for package, paths in {
        "monailabel.server": ["static/index.html", "resources/assistant/AGENTS.md"],
        "monailabel.viewers": [
            "resources/slicer_bridge.py",
            "resources/qupath/Backend.groovy",
            "resources/ohif/extension/src/speech.js",
            "resources/cvat/compose.yaml",
            "resources/desktop/Dockerfile",
        ],
        "sam2": ["configs/sam2.1_hiera_t512.yaml", "upstream.json"],
        "monailabel.nninteractive": ["worker.py", "worker.py.lock"],
    }.items():
        for path in paths:
            assert files(package).joinpath(path).read_bytes(), (package, path)

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torchvision.ops import nms

    from monailabel.sam.runtime import SamSegmenter
    from monailabel.sam.video import SamVideoTracker

    assert SamSegmenter and SamVideoTracker
    assert torch.version.cuda is not None, "The installed PyTorch build must include CUDA"
    assert nms(torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.ones(1), 0.5).tolist() == [0]
    # Resolve both model configurations and the MedSAM2-specific array predictor
    # with real upstream classes, without loading or downloading weights.
    for name in ("sam2", "medsam2"):
        config = OmegaConf.load(str(files("monailabel.sam").joinpath(f"resources/{name}.yaml")))
        config.model._target_ = "sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ"
        network = instantiate(config.model, _recursive_=True)
        assert network.fill_hole_area == 0
        del network
        gc.collect()

    from fastapi.testclient import TestClient

    from monailabel.server.app import create_app

    class NoChat:
        def complete(self, *args, **kwargs):
            raise AssertionError("Installation checks must not invoke a model")

    with tempfile.TemporaryDirectory(prefix="monailabel-installed-") as directory:
        root = Path(directory)
        os.environ["MONAILABEL_PRELOAD_MODELS"] = "0"
        os.environ["MONAILABEL_DATASETS_DIR"] = str(root / "datasets")
        os.environ["MONAILABEL_TOOLS_DIR"] = str(root / "tools")
        os.environ["MONAILABEL_ALLOWED_HOSTS"] = "testserver,127.0.0.1"
        password = secrets.token_urlsafe(24)
        with TestClient(create_app(root, chat_provider=NoChat())) as http:
            assert http.get("/").status_code == 200
            assert http.get("/static/app.js").status_code == 200
            assert http.get("/static/speech.js").status_code == 200
            assert http.get("/api/auth/status").json() == {"setup_required": True}
            assert (
                http.post(
                    "/api/auth/setup", json={"username": "owner", "password": password}
                ).status_code
                == 201
            )
            assert http.get("/api/health").json()["version"] == expected
            assert http.get("/openapi.json").json()["info"]["version"] == expected
            project = http.post("/api/projects", json={"name": "Install check"})
            assert project.status_code == 201, project.text
            pid = project.json()["id"]
            catalog = http.get(f"/api/projects/{pid}/dataset-templates").json()
            for identifier in (
                "tnbc-nuclei",
                "kvasir-instrument",
                "totalsegmentator-small",
                "totalsegmentator-mr",
                "prostate-mri",
            ):
                template = next(t for t in catalog if t["id"] == identifier)
                assert template["has_masks"] and template["citation"] and template["license_url"]
                assert template["labels_url"].startswith("https://")
            for role in ("reviewer", "annotator"):
                user = http.post("/api/auth/users", json={"username": role, "password": password})
                assert user.status_code == 201, user.text
                member = http.put(
                    f"/api/projects/{pid}/members",
                    json={"user_id": user.json()["id"], "roles": [role]},
                )
                assert member.status_code == 200, member.text
        # A new process/service can reopen the same persisted accounts and roles.
        with TestClient(create_app(root, chat_provider=NoChat())) as http:
            for role in ("reviewer", "annotator"):
                assert (
                    http.post(
                        "/api/auth/login", json={"username": role, "password": password}
                    ).status_code
                    == 200
                )
                assert http.get(f"/api/projects/{pid}/permissions").json() == {"roles": [role]}
                assert (
                    http.post(
                        "/api/auth/users", json={"username": "denied", "password": password}
                    ).status_code
                    == 403
                )
    print(f"MONAI Label {expected}: installed resources, SAM runtimes, API and persistence passed")


if __name__ == "__main__":
    main()
