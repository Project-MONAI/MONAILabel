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

"""Check a built image with disposable containers and a disposable persistent workspace."""

import argparse
import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(["docker", *args], check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--https", action="store_true", help="Verify generated HTTPS certificates.")
    args = parser.parse_args()
    run(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "--volume",
        f"{ROOT / 'scripts/check_installed.py'}:/check.py:ro",
        args.image,
        "-I",
        "/check.py",
    )
    for executable in ("docker", "node", "corepack", "ffmpeg"):
        flag = "-version" if executable == "ffmpeg" else "--version"
        run("run", "--rm", "--entrypoint", executable, args.image, flag, stdout=subprocess.DEVNULL)
    run("run", "--rm", "--entrypoint", "docker", args.image, "compose", "version")
    try:
        result = run(
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python",
            args.image,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "x=torch.ones(32, device='cuda'); assert (x+x).sum().item()==64; "
            "print('CUDA tensor operation passed:', torch.cuda.get_device_name(0))",
            capture_output=True,
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        # Hosted CI has no GPU device driver. Probe --gpus itself: a named
        # 'nvidia' runtime is not required when Docker's device hook is installed.
        if error.returncode == 125 and "could not select device driver" in error.stderr:
            print("Docker has no GPU device driver; CUDA execution check skipped.", flush=True)
        else:
            print(error.stdout, error.stderr)
            raise
    name = "monailabel-check-" + secrets.token_hex(6)
    # Access the real server through docker exec so bootstrap is genuinely loopback.
    probe = """import json, sys, ssl, urllib.request, http.cookiejar
body=json.load(sys.stdin)
handlers=[urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())]
if body['https']:
    handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile='/workspace/.tls/ca.crt')))
opener=urllib.request.build_opener(*handlers)
def call(path, value=None):
    req=urllib.request.Request(
        ('https' if body['https'] else 'http')+'://127.0.0.1:8000'+path,
        data=json.dumps(value).encode() if value else None,
        headers={'Content-Type':'application/json'})
    return json.load(opener.open(req, timeout=10))
assert call('/api/auth/status')['setup_required'] == body['setup']
call('/api/auth/setup' if body['setup'] else '/api/auth/login',
     {'username':'owner', 'password':body['password']})
if body['setup']:
    call('/api/projects', {'name':'Persistent project'})
assert [p['name'] for p in call('/api/projects')] == ['Persistent project']
print('Container startup and persistent account/project passed')
"""
    with tempfile.TemporaryDirectory(prefix="monailabel-docker-") as temporary:
        try:
            run(
                "run",
                "-d",
                "--name",
                name,
                "--volume",
                f"{temporary}:/workspace",
                "--env",
                "MONAILABEL_PRELOAD_MODELS=0",
                args.image,
                *(["--https"] if args.https else []),
                "--host",
                "0.0.0.0",
                "--assistant",
                "compatible",
                "--assistant-model",
                "fixture",
                "--assistant-base-url",
                "http://127.0.0.1:9/v1",
                stdout=subprocess.DEVNULL,
            )
            password = secrets.token_urlsafe(24)
            for setup in (True, False):
                deadline = time.monotonic() + 90
                while True:
                    result = subprocess.run(
                        [
                            "docker",
                            "exec",
                            name,
                            "python",
                            "-m",
                            "monailabel.server.healthcheck",
                        ],
                        capture_output=True,
                    )
                    if result.returncode == 0:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("Container server did not become ready")
                    time.sleep(1)
                run(
                    "exec",
                    "-i",
                    name,
                    "python",
                    "-c",
                    probe,
                    input=json.dumps({"setup": setup, "password": password, "https": args.https}),
                )
                if setup:
                    run("restart", name, stdout=subprocess.DEVNULL)
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL)
            # The image may run as root; remove only the disposable test directory.
            run(
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "--volume",
                f"{temporary}:/cleanup",
                args.image,
                "-c",
                "import pathlib, shutil; "
                "[(shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()) "
                "for p in pathlib.Path('/cleanup').iterdir()]",
            )


if __name__ == "__main__":
    main()
