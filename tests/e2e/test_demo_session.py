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

import re
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import VideoStack

from monailabel.server.demo_access import DemoAccess

pytestmark = pytest.mark.browser_e2e


def timed_visitor(stack, client):
    """Use a real non-admin invitation, only in the disposable test workspace."""
    stack.username = "countdown-visitor"
    visitor = client.post(
        "/api/auth/demo-users", {"username": stack.username, "password": stack.password}
    )
    client.post("/api/auth/login", {"username": stack.username, "password": stack.password})
    return visitor["id"]


def countdown(page, selector="#demo-countdown"):
    from playwright.sync_api import expect

    banner = page.locator(selector)
    expect(banner).to_be_visible(timeout=30000)
    expect(banner).to_contain_text("Demo time remaining:")
    expect(banner).to_contain_text("Public demo data only.")
    expect(banner).not_to_contain_text("recorded")
    hours, minutes, seconds = map(
        int, re.search(r"(\d\d):(\d\d):(\d\d)", banner.inner_text()).groups()
    )
    return hours * 3600 + minutes * 60 + seconds


def account_deadline(page):
    return page.evaluate("async () => (await (await fetch('/api/auth/session')).json()).expires_at")


def expire_visitor(stack, identifier):
    with stack.app.state.services.store.transaction() as session:
        lease = session.get(DemoAccess, identifier)
        session.update(
            lease.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
        )


def test_demo_notice_countdown_reload_and_expiry(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("MONAILABEL_DEMO_AUDIT", "1")
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(stack.url)
                expect(page.locator("#demo-notice")).to_contain_text("Research demo")
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                page.locator('nav [data-page="team"]').click()
                page.get_by_role("button", name="Create demo visitor", exact=True).click()
                dialog = page.locator("#dialog")
                dialog.get_by_label("Username", exact=True).fill("visitor")
                dialog.get_by_label("Password (at least 12 characters)", exact=True).fill(
                    "test-password-1234"
                )
                dialog.get_by_label("Demo duration").select_option("60")
                with page.expect_response("**/api/auth/demo-users") as created:
                    dialog.get_by_role("button", name="Create visitor", exact=True).click()
                visitor = created.value.json()
                assert created.value.status == 201
                page.locator("#logout").click()
                expect(page.locator("#auth")).to_be_visible()
                page.get_by_label("Username", exact=True).fill("visitor")
                page.get_by_label("Password", exact=True).fill("test-password-1234")
                page.locator("#login-button").click()
                expect(page.locator("#demo-countdown")).to_contain_text("Demo time remaining:")
                expect(page.locator("#demo-countdown")).to_contain_text("Public demo data only.")
                expect(page.locator("#demo-countdown")).not_to_contain_text("recorded")
                service = stack.app.state.services
                deadline = service.store.get(DemoAccess, visitor["id"]).expires_at
                page.reload()
                expect(page.locator("#demo-countdown")).to_be_visible()
                assert service.store.get(DemoAccess, visitor["id"]).expires_at == deadline
                page.screenshot(path=str(tmp_path / "demo-countdown.png"), full_page=True)
                with service.store.transaction() as session:
                    lease = session.get(DemoAccess, visitor["id"])
                    session.update(
                        lease.model_copy(
                            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
                        )
                    )
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                expect(page.locator("#auth")).to_be_visible()
                expect(page.locator("#auth-error")).to_contain_text("Demo access has ended")
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
