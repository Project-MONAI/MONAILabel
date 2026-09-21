import argparse
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.app import create_app
from monailabel.server.workspace import configure_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MONAI Label assistant service.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Workspace folder (default: MONAILABEL_DATA_DIR or ./workspace).",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Listen address (default: all IPv4 interfaces)."
    )
    parser.add_argument("--port", type=int, default=8000)
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
    try:
        values = {}
        for field in ("provider", "variant", "model", "base_url", "key_env", "gpu", "thinking"):
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
    app = create_app(configure_workspace(args.data_dir), coordinator=coordinator)
    app.state.direct_tls = bool(args.ssl_certfile)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
        ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None,
    )
