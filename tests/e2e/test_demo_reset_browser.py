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

"""An administrator must explicitly confirm a fresh demo; opening the dialog is safe."""

import httpx
import pytest
from conftest import VideoStack
from test_video_cvat import login_workspace

from monailabel.server.demo_access import DemoVisit

pytestmark = pytest.mark.browser_e2e


def test_confirmed_fresh_demo_keeps_administrator(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("MONAILABEL_DEMO_LIFECYCLE", "1")
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
            visitor = http.post(
                "/api/auth/demo-users",
                json={"username": "old-visitor", "password": "test-password-1234"},
            )
            visitor.raise_for_status()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                login_workspace(page, stack)
                page.locator('nav [data-page="team"]').click()
                page.get_by_role("button", name="Start a fresh demo", exact=True).click()
                dialog = page.locator("#dialog")
                expect(dialog).to_contain_text("remove 1 demo visitor")
                assert len(stack.app.state.services.store.list(DemoVisit)) == 1
                field = dialog.get_by_label("Type START FRESH DEMO to confirm")
                field.fill("incorrect")
                dialog.get_by_role("button", name="Reset demo visitors", exact=True).click()
                expect(dialog).to_be_visible()
                field.fill("START FRESH DEMO")
                page.screenshot(path=str(tmp_path / "fresh-demo-confirmation.png"), full_page=True)
                with page.expect_response("**/api/demo-access/reset") as result:
                    dialog.get_by_role("button", name="Reset demo visitors", exact=True).click()
                assert result.value.status == 200
                expect(page.locator("#workspace")).to_be_visible()
                expect(page.locator("#auth")).not_to_be_visible()
                assert stack.app.state.services.store.get(DemoVisit, visitor.json()["id"]).reset_at
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
