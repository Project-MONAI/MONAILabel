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

"""A first-time visitor can sign up without an administrator-created account."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import VideoStack

from monailabel.server.demo_access import DemoAccess

pytestmark = pytest.mark.browser_e2e


def test_first_visit_registration_isolation_busy_and_expiry(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    for key in (
        "MONAILABEL_DEMO_SELF_REGISTER",
        "MONAILABEL_DEMO_CACHE_ONLY",
        "MONAILABEL_DEMO_TRAINING_PREVIEW",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "2")
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            errors = []
            contexts = [
                browser.new_context(viewport={"width": 1400, "height": 1100}) for _ in range(3)
            ]
            try:
                pages = [context.new_page() for context in contexts]
                for index, page in enumerate(pages):
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(stack.url)
                    expect(
                        page.get_by_role("button", name="Create demo account", exact=True)
                    ).to_be_visible()
                    if index == 0:
                        page.screenshot(
                            path=str(tmp_path / "visitor-registration.png"), full_page=True
                        )
                    page.get_by_label("Username", exact=True).fill(f"browser-visitor-{index}")
                    page.get_by_label("Password", exact=True).fill("test-password-1234")
                    page.get_by_label("Confirm password", exact=True).fill("test-password-1234")
                    if index == 0:
                        page.get_by_label("Confirm password", exact=True).fill("different-password")
                        page.get_by_role("button", name="Create demo account", exact=True).click()
                        expect(page.locator("#auth-error")).to_contain_text(
                            "passwords do not match"
                        )
                        page.get_by_label("Confirm password", exact=True).fill("test-password-1234")
                    with page.expect_response("**/api/auth/register-demo") as response:
                        page.get_by_role("button", name="Create demo account", exact=True).click()
                    assert response.value.status == (201 if index < 2 else 429)
                    if index < 2:
                        expect(page.locator("#workspace")).to_be_visible()
                        expect(page.locator("#demo-countdown")).to_contain_text(
                            "Demo time remaining"
                        )
                        expect(
                            page.get_by_role("button", name="Load one sample CT", exact=True)
                        ).to_be_visible()
                        page.reload()
                        expect(page.locator("#demo-countdown")).to_be_visible()
                    else:
                        expect(page.locator("#auth-error")).to_contain_text("demo is busy")
                        expect(page.locator("#auth")).to_be_visible()
                        page.screenshot(path=str(tmp_path / "visitor-busy.png"), full_page=True)
                first_projects = contexts[0].request.get(stack.url + "/api/projects").json()
                second_projects = contexts[1].request.get(stack.url + "/api/projects").json()
                assert len(first_projects) == len(second_projects) == 1
                assert first_projects[0]["id"] != second_projects[0]["id"]
                assert (
                    contexts[1]
                    .request.get(stack.url + "/api/projects/" + first_projects[0]["id"])
                    .status
                    == 403
                )
                user = contexts[0].request.get(stack.url + "/api/auth/me").json()
                service = stack.app.state.services
                original = service.store.get(DemoAccess, user["id"])
                pages[0].locator("#logout").click()
                pages[0].get_by_role(
                    "button", name="Already have an account? Sign in", exact=True
                ).click()
                pages[0].get_by_label("Username", exact=True).fill("browser-visitor-0")
                pages[0].get_by_label("Password", exact=True).fill("test-password-1234")
                pages[0].get_by_role("button", name="Sign in", exact=True).click()
                expect(pages[0].locator("#workspace")).to_be_visible()
                assert service.store.get(DemoAccess, user["id"]) == original
                with service.store.transaction() as session:
                    session.update(
                        original.model_copy(
                            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
                        )
                    )
                pages[0].evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                expect(pages[0].locator("#auth-error")).to_contain_text("Demo access has ended")
                expect(pages[0].locator("#auth-error")).to_contain_text("new demo account")
                assert not errors
            finally:
                for context in contexts:
                    context.close()
                browser.close()
    finally:
        stack.stop_workspace()
