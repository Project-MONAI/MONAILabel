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

"""Real local assistant, public CT import and VISTA3D annotation in a disposable workspace.

Requires approved installation and cached public data/models. Does not configure
hosted models or accept annotations. Checks execution, not medical correctness.
"""

import argparse
import json
import secrets
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from monailabel.client.client import Client
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.app import create_app
from monailabel.server.coordinator_runtime import CoordinatorRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = CoordinatorRuntime(CoordinatorConfig(gpu=args.gpu, timeout=180))
    provider.prepare()
    if provider.state != "ready":
        raise RuntimeError(provider.detail)
    report = {"hosted_api_calls": 0, "clinical_accuracy_evaluated": False, "steps": []}
    try:
        with (
            tempfile.TemporaryDirectory(prefix="monailabel-private-acceptance-") as temp,
            TestClient(create_app(Path(temp), chat_provider=provider)) as http,
        ):
            client = Client(http=http)
            client.post(
                "/api/auth/setup",
                {"username": "disposable-tester", "password": secrets.token_urlsafe(32)},
            )
            project_id = conversation_id = None
            prompts = [
                'Create a project called "Private radiology test".',
                "Import 3 Decathlon Spleen images without labels.",
                "Annotate spleen in 1 image with VISTA3D; submit for review.",
            ]
            for prompt in prompts:
                started = time.monotonic()
                print(prompt, flush=True)
                reply = client.post(
                    "/api/assistant",
                    {
                        "project_id": project_id,
                        "conversation_id": conversation_id,
                        "message": prompt,
                    },
                )
                data = reply["data"]
                if reply["job_id"]:
                    data = client.wait(reply["job_id"], timeout=900)
                    if data.get("failed"):
                        raise RuntimeError("The annotation job reported failed samples.")
                if project_id is None:
                    project_id = data["project"]["id"]
                    conversation_id = None
                else:
                    conversation_id = reply["conversation_id"]
                report["steps"].append(
                    {
                        "prompt": prompt,
                        "tools": reply["tools"],
                        "message": reply["message"],
                        "seconds": round(time.monotonic() - started, 2),
                    }
                )
            assets = client.get(f"/api/projects/{project_id}/assets")
            assert len(assets) == 3, "The requested three public samples were not imported."
            submitted = [asset for asset in assets if asset["annotation_id"]]
            assert len(submitted) == 1 and submitted[0]["revision"] == 1, (
                "Expected one pending annotation; inspect the recorded assistant reply."
            )
            assert client.get(f"/api/projects/{project_id}/decisions") == []
            report.update(imported_samples=3, pending_annotations=1, passed=True)
    except Exception as exc:
        report.update(passed=False, error=str(exc) or type(exc).__name__)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
