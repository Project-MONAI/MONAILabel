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

"""Real browser-hosted native viewers share a visitor's persisted deadline."""

import io
import re

import pytest
from PIL import Image
from test_all_structures_browser import prepare
from test_browser_desktop import desktop_stack, fitted_desktop, observed  # noqa: F401
from test_demo_session import account_deadline, countdown, expire_visitor, timed_visitor
from test_video_cvat import login_workspace

from monailabel.core.models import Role
from monailabel.server.desktops.models import DesktopSession


@pytest.fixture(autouse=True)
def enable_timed_cleanup(monkeypatch):
    monkeypatch.setenv("MONAILABEL_DEMO_LIFECYCLE", "1")


@pytest.mark.desktop_e2e
@pytest.mark.parametrize("viewer", ["slicer", "qupath"])
def test_native_wrapper_countdown(desktop_stack, viewer, monkeypatch):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright

    stack, client = desktop_stack
    project, asset, _ = prepare(stack, client, monkeypatch)
    if viewer == "qupath":
        image = io.BytesIO()
        Image.new("RGB", (128, 128), (210, 160, 185)).save(image, format="PNG")
        response = client.http.post(
            f"/api/projects/{project['id']}/assets/upload",
            params={"name": "Clock test slide.png", "group_id": "clock-slide"},
            content=image.getvalue(),
        )
        response.raise_for_status()
        asset = response.json()
    identifier = timed_visitor(stack, client)
    stack.app.state.services.auth.add_member(project["id"], identifier, [Role.MANAGER])
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            deadline = account_deadline(page)
            with page.expect_popup() as opened:
                page.get_by_role(
                    "button", name="Slicer" if viewer == "slicer" else "QuPath", exact=True
                ).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=180000)
            desktop_id = desktop.url.rsplit("/", 1)[-1]
            expect(desktop.frame_locator("#display").locator("html")).to_have_class(
                re.compile("noVNC_connected"), timeout=30000
            )
            runtime = stack.app.state.services.desktops.runtime
            observed(
                runtime.root / desktop_id / "observed.json",
                lambda value: value["asset_id"] == asset["id"],
                desktop,
            )
            fitted_desktop(desktop)
            assert abs(countdown(page) - countdown(desktop)) <= 2
            assert account_deadline(desktop) == deadline
            desktop.screenshot(path=str(stack.artifacts / f"{viewer}-countdown.png"))
            desktop.reload()
            fitted_desktop(desktop)
            countdown(desktop)
            assert account_deadline(desktop) == deadline
            expire_visitor(stack, identifier)
            desktop.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
            expect(desktop.locator("#auth")).to_be_visible(timeout=30000)
            assert desktop.evaluate("async () => (await fetch('/api/projects')).status") == 401
            for _ in range(100):
                if stack.app.state.services.store.get(DesktopSession, desktop_id).ended:
                    break
                desktop.wait_for_timeout(100)
            assert stack.app.state.services.store.get(DesktopSession, desktop_id).ended
        finally:
            browser.close()
