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

"""One opt-in paid GPT annotation call on a generated circle, offline by default.

Uses the real polygon adapter and the exact approved demo connection. No running
workspace is opened. The private output marker prevents an accidental paid repeat.
This synthetic connectivity/geometry check is not a medical accuracy evaluation.
"""

import argparse
import json
import logging
import math
import os
import resource
import stat
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from demo_openai_run import read_key
from PIL import Image

from monailabel.core.models import Label, ModelRecord
from monailabel.providers import remote
from monailabel.server.demo_gpt import MODELS, configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="demo-gpt-sol")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.execute and args.key_file is None:
        parser.error("--execute requires --key-file")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = args.output.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit("Use an owner-only private output directory.")
    key = read_key(args.key_file) if args.execute else "sk-offline-not-a-real-key"
    if args.execute:
        fd = os.open(args.output / "paid-started", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    name, identifier = MODELS[args.model]
    config = configuration(identifier)
    report = {
        "model": identifier,
        "live": args.execute,
        "requests": 0,
        "passed": False,
        "live_workspace_accessed": False,
        "patient_data_used": False,
        "input": "generated white circle on black background",
        "automatic_retries": 0,
        "max_output_tokens": config["max_output_tokens"],
    }
    y, x = np.mgrid[:256, :256]
    truth = (x - 127.5) ** 2 + (y - 127.5) ** 2 <= 64**2
    image = np.repeat(truth[:, :, None], 3, axis=2).astype(np.float32)
    labels = [Label(id=0, name="Background"), Label(id=1, name="White circle")]
    model = ModelRecord(
        project_id="synthetic-check",
        name=name,
        provider="openai-polygons",
        label_ids=[0, 1],
        preset=args.model,
        read_only=True,
        config=config,
    )

    def guard(request):
        body = json.loads(request.content)
        if (
            request.method != "POST"
            or str(request.url) != config["url"]
            or report["requests"]
            or body.get("model") != identifier
            or body.get("max_output_tokens") != config["max_output_tokens"]
            or body.get("store") is not False
            or len(request.content) > 40000
        ):
            raise RuntimeError("Request outside the approved single synthetic test.")
        report["requests"] += 1

    def observe(response):
        response.read()
        report["http_status"] = response.status_code
        if response.status_code == 200:
            usage = response.json().get("usage", {})
            report["usage"] = {
                k: usage[k]
                for k in ("input_tokens", "output_tokens", "total_tokens")
                if isinstance(usage.get(k), int)
            }

    def offline(request):
        points = [
            {"x": 128 + 64 * math.cos(a), "y": 128 + 64 * math.sin(a)}
            for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)
        ]
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {"polygons": [{"label_id": 1, "points": points}]}
                                ),
                            }
                        ],
                    }
                ],
            },
        )

    original_client = httpx.Client

    def client(**kwargs):
        transport = httpx.HTTPTransport(retries=0) if args.execute else httpx.MockTransport(offline)
        return original_client(
            **kwargs,
            trust_env=False,
            transport=transport,
            event_hooks={"request": [guard], "response": [observe]},
        )

    started = time.monotonic()
    try:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": key}),
            patch.object(remote.httpx, "Client", side_effect=client),
        ):
            prediction = remote.RemoteSegmenter("openai-polygons", lambda *args: "").predict(
                image,
                labels,
                "Segment the white circle. Trace its boundary using roughly 24 polygon points. "
                "Do not outline the black background.",
                model,
            )
        mask = prediction.mask
        predicted = mask == 1
        dice = 2 * int(np.count_nonzero(truth & predicted)) / int(truth.sum() + predicted.sum())
        valid = mask.shape == (256, 256) and set(np.unique(mask)) <= {0, 1}
        report.update(
            valid_geometry=bool(valid),
            synthetic_dice=round(dice, 6),
            passed=bool(valid and dice >= 0.85),
        )
        Image.fromarray((image * 255).astype(np.uint8)).save(args.output / "input.png")
        Image.fromarray((predicted * 255).astype(np.uint8)).save(args.output / "mask.png")
    except Exception as error:
        report["error_type"] = type(error).__name__  # Never persist upstream bodies or keys.
    report["seconds"] = round(time.monotonic() - started, 2)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
