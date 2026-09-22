"""Opt-in infrastructure, isolated from user workspaces and CVAT deployments."""

import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from chat_fixture import ScriptedChat

from monailabel.server.app import create_app

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "packages/viewers/src/monailabel/viewers/resources/cvat/compose.yaml"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class VideoStack:
    root: Path
    artifacts: Path
    compose_project: str = field(
        default_factory=lambda: "monailabel-video-e2e-" + secrets.token_hex(6)
    )
    username: str = field(default="e2e-user", repr=False)
    password: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    token: str = field(default="", repr=False)
    api_port: int = field(default_factory=free_port)
    ui_port: int = field(default_factory=free_port)
    server: object = field(default=None, repr=False)
    thread: object = field(default=None, repr=False)

    @property
    def cvat_url(self):
        return f"http://localhost:{self.ui_port}"

    def compose(self, *args, input=None, timeout=180, check=True):
        env = os.environ | {
            "MONAILABEL_CVAT_API_PORT": str(self.api_port),
            "MONAILABEL_CVAT_UI_PORT": str(self.ui_port),
        }
        result = subprocess.run(
            ["docker", "compose", "-p", self.compose_project, "-f", str(COMPOSE), *args],
            input=input,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        if check and result.returncode:
            # Provisioning input can contain disposable secrets; never include it in failures.
            log = result.stdout + result.stderr
            for value in (self.password, self.token):
                if value:
                    log = log.replace(value, "[redacted]")
            (self.artifacts / "compose-error.log").write_text(log)
            raise RuntimeError(f"CVAT {args[0]} failed; see the test's compose-error.log.")
        return result

    def start_cvat(self):
        self.compose("up", "-d", timeout=600)
        deadline = time.monotonic() + 180
        with httpx.Client(base_url=self.cvat_url, timeout=5) as http:
            while time.monotonic() < deadline:
                try:
                    response = http.get("/api/server/about")
                    if response.status_code == 200:
                        assert response.json()["version"] == "2.76.0"
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Disposable CVAT did not become ready within three minutes.")
            code = (
                "from django.contrib.auth import get_user_model\n"
                f"get_user_model().objects.create_superuser("
                f"{self.username!r}, '', {self.password!r})\n"
            )
            self.compose("exec", "-T", "server", "python", "manage.py", "shell", input=code)
            response = http.post(
                "/api/auth/login", json={"username": self.username, "password": self.password}
            )
            response.raise_for_status()
            self.token = response.json()["key"]

    def start_workspace(self):
        assert self.server is None
        # The socket stays reserved until Uvicorn takes ownership.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.app = create_app(self.root / "workspace", chat_provider=ScriptedChat())
        self.server = uvicorn.Server(uvicorn.Config(self.app, access_log=False, log_level="error"))
        self.thread = threading.Thread(target=lambda: self.server.run(sockets=[sock]), daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if not self.thread.is_alive():
                break
            time.sleep(0.05)
        raise RuntimeError("Disposable MONAI Label server did not start.")

    def stop_workspace(self):
        if self.server:
            self.server.should_exit = True
            self.thread.join(timeout=30)
            assert not self.thread.is_alive(), "Workspace server did not shut down"
            self.server = None

    def save_logs(self):
        result = self.compose("logs", "--no-color", "--tail", "150", check=False)
        log = result.stdout + result.stderr
        for value in (self.password, self.token):
            if value:
                log = log.replace(value, "[redacted]")
        (self.artifacts / "services.log").write_text(log)


@pytest.fixture(scope="module")
def video_stack(tmp_path_factory):
    for executable in ("docker", "ffmpeg", "ffprobe"):
        assert shutil.which(executable), f"Install {executable} to run --video-e2e."
    directory = tmp_path_factory.mktemp("video-e2e")
    artifacts = ROOT / "test-results" / ("video-" + secrets.token_hex(6))
    artifacts.mkdir(parents=True, mode=0o700)
    stack = VideoStack(directory, artifacts)
    if stack.api_port == stack.ui_port:
        stack.ui_port = free_port()
    print(f"\nVideo E2E artifacts: {artifacts}", flush=True)
    with patch.dict(
        os.environ,
        {
            "MONAILABEL_PRELOAD_MODELS": "0",
            "MONAILABEL_CVAT_URL": stack.cvat_url,
            "MONAILABEL_CVAT_PUBLIC_URL": stack.cvat_url,
        },
    ):
        try:
            stack.start_cvat()
            with patch.dict(os.environ, {"MONAILABEL_CVAT_TOKEN": stack.token}):
                stack.start_workspace()
                with httpx.Client(base_url=stack.url) as http:
                    response = http.post(
                        "/api/auth/setup",
                        json={"username": stack.username, "password": stack.password},
                    )
                    response.raise_for_status()
                yield stack
        finally:
            try:
                stack.stop_workspace()
                stack.save_logs()
            finally:
                # Only the unpredictable project created by this fixture is ever removed.
                try:
                    stack.compose("down", "--volumes", "--remove-orphans", timeout=120)
                finally:
                    if stack.server is None:
                        shutil.rmtree(directory)


@pytest.fixture
def synthetic_clip(video_stack):
    clip = video_stack.root / "synthetic-instruments.mp4"
    if not clip.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=10",
                "-frames:v",
                "6",
                "-vf",
                r"setpts=if(lt(N\,3)\,N/(10*TB)\,(N+2)/(10*TB))",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    return clip


@pytest.fixture
def browser_session(video_stack, request):
    # Imported only for explicit E2E runs; ordinary pytest needs neither Playwright nor Docker.
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP cvat.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page_errors = []
        context.on(
            "page", lambda page: page.on("pageerror", lambda error: page_errors.append(str(error)))
        )
        try:
            yield context, page_errors
        finally:
            try:
                for index, page in enumerate(context.pages):
                    if not page.is_closed():
                        try:
                            page.screenshot(
                                path=video_stack.artifacts / f"{request.node.name}-{index}.png",
                                full_page=True,
                                timeout=10000,
                            )
                        except Error as error:
                            # A crashed page must not hide the original test failure.
                            (
                                video_stack.artifacts / f"{request.node.name}-screenshot-error.txt"
                            ).write_text(str(error))
                (video_stack.artifacts / f"{request.node.name}-browser-errors.json").write_text(
                    json.dumps(page_errors, indent=2)
                )
            finally:
                context.close()
                browser.close()


@contextmanager
def authenticated_api(stack, *, cvat=False):
    with httpx.Client(base_url=stack.cvat_url if cvat else stack.url, timeout=60) as http:
        if cvat:
            http.headers["Authorization"] = "Token " + stack.token
        else:
            response = http.post(
                "/api/auth/login", json={"username": stack.username, "password": stack.password}
            )
            response.raise_for_status()
        yield http


@pytest.fixture
def video_http(video_stack):
    with authenticated_api(video_stack) as http:
        yield http


@pytest.fixture
def cvat_http(video_stack):
    with authenticated_api(video_stack, cvat=True) as http:
        yield http


@pytest.fixture
def detection_endpoint():
    """Deterministic vision HTTP boundary; CVAT, source decoding and SAM remain real."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace

    state = SimpleNamespace(calls=[], box=[420, 390, 600, 570], status="found", polygon=None)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(payload)
            content = {"status": state.status, "box": state.box}
            if payload["response_format"]["json_schema"]["name"] == "segmentation":
                prompt = json.loads(payload["messages"][-1]["content"][0]["text"])
                label = next(label["id"] for label in prompt["labels"] if label["id"])
                content = {
                    "polygons": [
                        {
                            "label_id": label,
                            "points": [{"x": x, "y": y} for x, y in state.polygon],
                        }
                    ]
                }
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(content),
                            },
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
