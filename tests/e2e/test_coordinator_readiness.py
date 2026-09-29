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

"""Startup status gates chat and refreshes without a project or a page reload."""

import pytest
from conftest import VideoStack

from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.coordinator_runtime import CoordinatorRuntime

pytestmark = pytest.mark.browser_e2e


@pytest.mark.parametrize("provider", ["local", "openai"])
def test_chat_waits_for_readiness_and_preserves_the_draft(tmp_path, provider):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    runtime = CoordinatorRuntime(
        CoordinatorConfig(provider=provider, model="gpt-6-astra" if provider == "openai" else None)
    )
    runtime.detail = "Checking coordinator tool calling."
    stack.app.state.services.coordinator = runtime
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                errors = []
                submitted = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "request",
                    lambda request: (
                        submitted.append(request)
                        if request.url.endswith("/api/assistant")
                        else None
                    ),
                )
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                status = page.locator(".chat-footnote")
                send = page.get_by_role("button", name="Send message", exact=True)
                draft = page.locator("#chat-input")
                prompt = "Create a project called Liver follow-up"
                expect(status).to_have_text("Checking coordinator tool calling.")
                expect(send).to_be_disabled()
                draft.fill(prompt)
                draft.press("Enter")
                expect(draft).to_have_value(prompt)
                assert not submitted

                runtime.state = "error"
                runtime.detail = "Coordinator did not pass its tool-calling readiness check."
                expect(status).to_have_text(runtime.detail, timeout=10000)
                expect(send).to_be_disabled()
                expect(draft).to_have_value(prompt)

                runtime.state = "ready"
                runtime.detail = "Local coordinator is ready."
                expect(status).to_have_text("Assistant ready", timeout=10000)
                expect(send).to_be_enabled()
                expect(draft).to_have_value(prompt)
                send.click()
                expect(page.locator("#project-select option:checked")).to_have_text(
                    "Liver follow-up"
                )
                assert len(submitted) == 1
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
