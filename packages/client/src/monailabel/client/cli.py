import argparse
import base64
import getpass
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path

import httpx

from monailabel.client.client import Client
from monailabel.client.demo import run_demo
from monailabel.core.errors import DomainError
from monailabel.viewers.manager import ViewerManager


def main() -> None:
    parser = argparse.ArgumentParser(description="MONAI Label workflows and viewer setup")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the synthetic NIfTI learning/handoff verification")
    commands.add_parser(
        "seed", help="Create an unreviewed synthetic project for manual exploration"
    )
    commands.add_parser("projects")
    login = commands.add_parser("login", help="Sign in and save a local session token")
    login.add_argument("--username", required=True)
    login.add_argument("--setup", action="store_true", help="Create the first administrator")
    chat = commands.add_parser("chat")
    chat.add_argument("project_id")
    chat.add_argument("message")
    for name in ("asset-id", "model-id", "baseline-id", "snapshot-id", "evaluation-id"):
        chat.add_argument(f"--{name}")
    chat.add_argument(
        "--no-launch", action="store_true", help="Prepare viewer without opening its GUI"
    )
    importer = commands.add_parser("import")
    importer.add_argument("project_id")
    importer.add_argument("path", type=Path)
    importer.add_argument("--group", required=True)
    importer.add_argument("--split", choices=["train", "validation", "pool"], default="pool")
    request = commands.add_parser("request", help="Call a documented API endpoint")
    request.add_argument("method", choices=["GET", "POST", "PUT"])
    request.add_argument("path")
    request.add_argument("--json", type=Path, help="Read a request body from a JSON file")
    wait = commands.add_parser("wait")
    wait.add_argument("job_id")
    viewer = commands.add_parser("viewer")
    viewer.add_argument("name", default="slicer", nargs="?")
    viewer.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    try:
        with Client(args.url) as client:
            if args.command == "login":
                client.login(args.username, getpass.getpass("Password: "), setup=args.setup)
                result = {"signed_in": args.username}
            elif args.command == "demo":
                result = run_demo(client)
            elif args.command == "seed":
                result = client.post("/api/demo")
            elif args.command == "projects":
                result = client.get("/api/projects")
            elif args.command == "wait":
                result = client.wait(args.job_id)
            elif args.command == "request":
                data = json.loads(args.json.read_text()) if args.json else None
                result = client.request(args.method, args.path, data)
            elif args.command == "import":
                result = client.post(
                    f"/api/projects/{args.project_id}/assets",
                    {
                        "name": args.path.name,
                        "group_id": args.group,
                        "split": args.split,
                        "image_base64": base64.b64encode(args.path.read_bytes()).decode(),
                    },
                )
            elif args.command == "viewer":
                if args.name.casefold() == "ohif":
                    from monailabel.viewers.ohif import OhifManager

                    result = {"viewer": "ohif", "dist": str(OhifManager().ensure())}
                else:
                    result = asdict(
                        ViewerManager().ensure(args.name, download=not args.no_download)
                    )
            else:
                context = {
                    name: getattr(args, name)
                    for name in (
                        "asset_id",
                        "model_id",
                        "baseline_id",
                        "snapshot_id",
                        "evaluation_id",
                    )
                    if getattr(args, name)
                }
                result = client.post(
                    f"/api/projects/{args.project_id}/assistant",
                    {
                        "message": args.message,
                        "context": context,
                    },
                )
                if (
                    result["data"].get("client_action") == "use_viewer"
                    and result["data"]["viewer"] == "ohif"
                ):
                    if not args.asset_id:
                        raise DomainError("Select a DICOM sample with --asset-id.")
                    job = client.post(f"/api/assets/{args.asset_id}/viewer?name=ohif")
                    launch = client.wait(job["id"], timeout=1800)
                    result["url"] = args.url.rstrip("/") + launch["url"]
                    if not args.no_launch:
                        webbrowser.open(result["url"])
                elif result["data"].get("client_action") == "use_viewer":
                    manager = ViewerManager()
                    installation = manager.ensure(result["data"]["viewer"])
                    result["viewer"] = asdict(installation)
                    if not args.no_launch:
                        token = client.post("/api/auth/token")["token"]
                        secret_env = {
                            name
                            for model in client.get(f"/api/projects/{args.project_id}/models")
                            if isinstance(name := model["config"].get("token_env"), str)
                        }
                        result["launch"] = manager.launch(
                            installation,
                            args.url,
                            args.project_id,
                            asset_id=args.asset_id,
                            token=token,
                            secret_env=secret_env,
                        )
                elif result.get("job_id"):
                    result["result"] = client.wait(result["job_id"])
            print(json.dumps(result, indent=2))
    except (DomainError, RuntimeError, TimeoutError, httpx.HTTPError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
