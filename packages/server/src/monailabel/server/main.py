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

import argparse
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from monailabel.core import __version__
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.app import create_app
from monailabel.server.tls import ensure_local_certificate
from monailabel.server.workspace import configure_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MONAI Label assistant service.")
    parser.add_argument("--version", action="version", version=f"MONAI Label {__version__}")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Workspace folder (default: MONAILABEL_DATA_DIR or ./workspace).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Listen address (default: localhost only)."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--https",
        action="store_true",
        help="Generate and reuse a local HTTPS certificate in the workspace.",
    )
    parser.add_argument("--ssl-certfile", type=Path, help="TLS certificate for HTTPS access.")
    parser.add_argument(
        "--ssl-keyfile", type=Path, help="Private key matching the TLS certificate."
    )
    parser.add_argument(
        "--assistant",
        choices=["local", "openai", "anthropic", "gemini", "compatible"],
        default=None,
        help="Prompt coordinator provider; local Nemotron by default.",
    )
    parser.add_argument("--assistant-model", help="Hosted/existing endpoint model ID.")
    parser.add_argument(
        "--assistant-base-url", help="Hosted/existing API base URL, including /v1 when required."
    )
    parser.add_argument(
        "--assistant-key-env",
        help=(
            "Environment variable containing the coordinator API key; optional "
            "for unauthenticated endpoints."
        ),
    )
    parser.add_argument(
        "--assistant-gpu", help="GPU index/UUID for the local coordinator (default 0)."
    )
    parser.add_argument(
        "--assistant-gpu-memory-utilization",
        type=float,
        help="Local serving memory fraction (0.05–0.95); defaults depend on model and platform.",
    )
    parser.add_argument(
        "--assistant-variant",
        choices=["4b", "9b", "lightning"],
        help="Local Nemotron size; default lightning.",
    )
    parser.add_argument(
        "--assistant-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable local reasoning; enabled by default.",
    )
    args = parser.parse_args()
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("HTTPS requires both --ssl-certfile and --ssl-keyfile.")
    if args.https and args.ssl_certfile:
        parser.error("Use --https or your own --ssl-certfile/--ssl-keyfile pair.")
    try:
        values = {}
        for field in (
            "provider",
            "variant",
            "model",
            "base_url",
            "key_env",
            "gpu",
            "thinking",
            "gpu_memory_utilization",
        ):
            value = getattr(args, "assistant" if field == "provider" else "assistant_" + field)
            if value is not None:
                values[field] = value
        coordinator = CoordinatorConfig.from_env(**values)
    except ValidationError:
        parser.error(
            "Invalid coordinator configuration. Hosted providers need --assistant-model; "
            "compatible also needs --assistant-base-url. Pass key environment "
            "names, never secret values."
        )
    directory = configure_workspace(args.data_dir)
    ca_certificate = None
    if args.https:
        try:
            local = ensure_local_certificate(directory, args.host)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot prepare local HTTPS: {error}")
        args.ssl_certfile, args.ssl_keyfile = local.certificate, local.key
        ca_certificate = local.authority.read_text()
        print(f"HTTPS enabled. Trust {local.authority} on each browser device to use voice input.")
        print("Copy only ca.crt; keep both .key files private. See docs/viewers.md#https.")
    app = create_app(directory, coordinator=coordinator)
    app.state.direct_tls = bool(args.ssl_certfile)
    app.state.local_ca_certificate = ca_certificate
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
        ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None,
    )
