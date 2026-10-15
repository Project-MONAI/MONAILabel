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

"""Opt-in staged native-viewer capacity check; no production workspace or API keys."""

import json
import os
import subprocess
import threading
import time

import httpx
import pytest
from test_browser_desktop import desktop_stack as desktop_stack
from test_browser_desktop import fitted_desktop, observed
from test_prepared_visitor_viewers import sign_in

from monailabel.client.client import Client


@pytest.fixture(autouse=True)
def capacity_configuration(isolated_presets, monkeypatch):
    cache = os.environ.get("MONAILABEL_TEST_STARTER_CACHE")
    if not cache:
        pytest.skip("Requires prepared public sample and native-viewer caches")
    monkeypatch.setenv("MONAILABEL_DATASETS_DIR", cache)
    for key in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + key, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "5")
    monkeypatch.setenv("MONAILABEL_DESKTOP_LIMIT", "5")
    monkeypatch.setenv("MONAILABEL_DESKTOP_MEMORY", "4g")
    monkeypatch.setenv("MONAILABEL_DESKTOP_CPUS", "1")


@pytest.mark.desktop_e2e
def test_four_then_five_real_viewers(desktop_stack):
    from playwright.sync_api import sync_playwright

    stack, _ = desktop_stack
    runtime = stack.app.state.services.desktops.runtime
    report = {
        "sample": "three public CT Slicer viewers and two public pathology QuPath viewers",
        "desktop_memory_limit": "4g",
        "desktop_cpu_limit": 1,
        "stages": [],
        "live_health": [],
    }
    done = threading.Event()
    unhealthy = threading.Event()

    def watch():
        failures = 0
        with httpx.Client(timeout=5, trust_env=False) as live:
            while not done.is_set():
                start = time.monotonic()
                try:
                    response = live.get("http://127.0.0.1:8000/api/auth/status")
                    response.raise_for_status()
                    failures = 0
                    report["live_health"].append(
                        {"status": response.status_code, "ms": (time.monotonic() - start) * 1000}
                    )
                except Exception as error:
                    failures += 1
                    report["live_health"].append({"error": type(error).__name__})
                if failures >= 2:
                    unhealthy.set()
                done.wait(3)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    clients, contexts, desktops, identifiers = [], [], [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
            )
            try:
                for index, viewer in enumerate(("slicer", "qupath", "slicer", "qupath", "slicer")):
                    assert not unhealthy.is_set(), "Stop staging on failed live health checks"
                    client = Client(
                        http=httpx.Client(
                            base_url=stack.url.replace("desktop.test", "127.0.0.1"),
                            timeout=30,
                            trust_env=False,
                        )
                    )
                    clients.append(client)
                    credentials = {
                        "username": f"capacity-viewer-{index}",
                        "password": stack.password,
                    }
                    client.post("/api/auth/signup", credentials)
                    client.post("/api/auth/login", credentials)
                    project = client.get("/api/projects")[0]["id"]
                    for job in client.get(f"/api/projects/{project}/jobs"):
                        client.wait(job["id"], timeout=180)
                    if viewer == "qupath":
                        job = client.post("/api/demo-access/starter", {"example": "pathology"})
                        client.wait(job["id"], timeout=180)
                    kind = "volume3d" if viewer == "slicer" else "image2d"
                    asset = next(
                        a
                        for a in client.get(f"/api/projects/{project}/assets")
                        if a["kind"] == kind
                    )
                    context = browser.new_context(viewport={"width": 1280, "height": 900})
                    contexts.append(context)
                    page = context.new_page()
                    sign_in(page, stack, credentials, project)
                    with page.expect_popup() as opened:
                        page.get_by_role(
                            "button", name="Slicer" if viewer == "slicer" else "QuPath", exact=True
                        ).first.click()
                    desktop = opened.value
                    desktops.append(desktop)
                    desktop.wait_for_url("**/desktop/*", timeout=180000)
                    identifier = desktop.url.rsplit("/", 1)[-1]
                    identifiers.append(identifier)
                    fitted_desktop(desktop)
                    observed(
                        runtime.root / identifier / "observed.json",
                        lambda value, expected=asset["id"]: value["asset_id"] == expected,
                        desktop,
                        timeout=180,
                    )
                    if len(desktops) in {4, 5}:
                        start = time.monotonic()
                        for old in desktops:
                            old.bring_to_front()
                            fitted_desktop(old)
                            assert (
                                old.evaluate("async () => (await fetch('/api/projects')).status")
                                == 200
                            )
                            old.mouse.move(500, 400)
                            old.wait_for_timeout(400)
                        for connection in clients:
                            assert len(connection.get("/api/projects")) == 1
                        stats = subprocess.check_output(
                            [
                                "docker",
                                "stats",
                                "--no-stream",
                                "--format",
                                "{{json .}}",
                                *(runtime.name(item) for item in identifiers),
                            ],
                            text=True,
                            timeout=30,
                        )
                        report["stages"].append(
                            {
                                "viewers": len(desktops),
                                "interaction_seconds": time.monotonic() - start,
                                "containers": [json.loads(line) for line in stats.splitlines()],
                            }
                        )
                        desktop.screenshot(
                            path=stack.artifacts / f"capacity-{len(desktops)}-viewers.png"
                        )
                        assert not unhealthy.is_set()
                report["ok"] = True
            finally:
                for context in contexts:
                    context.close()
                browser.close()
    finally:
        for client in clients:
            client.http.close()
        done.set()
        watcher.join(timeout=10)
        (stack.artifacts / "capacity-viewers.json").write_text(json.dumps(report, indent=2))
