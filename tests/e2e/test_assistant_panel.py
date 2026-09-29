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

"""Readable assistant Markdown and persistent, accessible panel resizing."""

import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e


def test_assistant_markdown_and_resizing(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(error.stack))
                page.route(
                    "**/api/assistant/status",
                    lambda route: route.fulfill(json={"state": "ready", "message": "Ready"}),
                )
                markdown = (
                    "**VISTA3D training defaults**\n\n"
                    "| Parameter | Default | Notes |\n| --- | --- | --- |\n"
                    "| Epochs | 5 | Complete training epochs |\n"
                    "| Learning rate | `0.00005` | Initial optimizer learning rate |\n\n"
                    "- Review training masks.\n- Inspect held-out scores.\n\n"
                    "Line one\nLine two\n\n```text\nTrain the selected model.\n```\n\n"
                    '[Documentation](https://example.org/docs)\n\n<img src=x onerror="alert(1)">'
                )
                page.route(
                    "**/api/assistant",
                    lambda route: route.fulfill(
                        json={"message": markdown, "tools": [], "data": {}, "job_id": None}
                    ),
                )
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                page.locator("#chat-input").fill("List VISTA3D training defaults")
                page.get_by_role("button", name="Send message", exact=True).click()
                reply = page.locator(".assistant-message .chat-content").last
                expect(reply.locator("table")).to_be_visible()
                expect(reply.locator("strong")).to_have_text("VISTA3D training defaults")
                expect(reply.locator("li")).to_have_count(2)
                expect(reply.locator("pre code")).to_have_text("Train the selected model.\n")
                expect(reply.locator("img, script")).to_have_count(0)
                assert (
                    reply.locator("tbody tr").first.evaluate("row => getComputedStyle(row).display")
                    == "block"
                )
                panel = page.locator("#assistant-panel")
                assert reply.evaluate("node => node.scrollWidth <= node.clientWidth")
                page.screenshot(path=str(tmp_path / "assistant-narrow.png"))
                handle = page.get_by_role("separator", name="Resize workspace assistant")
                before = panel.bounding_box()["width"]
                box = handle.bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, 400)
                page.mouse.down()
                page.mouse.move(box["x"] - 245, 400, steps=10)
                page.mouse.up()
                after = panel.bounding_box()["width"]
                assert after > before + 200
                assert (
                    reply.locator("tbody tr").first.evaluate("row => getComputedStyle(row).display")
                    == "table-row"
                )
                assert reply.evaluate("node => node.scrollWidth <= node.clientWidth")
                page.screenshot(path=str(tmp_path / "assistant-wide.png"))
                handle.focus()
                page.keyboard.press("ArrowRight")
                preferred = panel.bounding_box()["width"]
                assert preferred == after - 32
                page.reload()
                expect(page.locator("#workspace")).to_be_visible()
                assert panel.bounding_box()["width"] == preferred
                page.get_by_role("button", name="Hide assistant").click()
                expect(panel).not_to_be_visible()
                page.locator("#toggle-assistant").click()
                expect(panel).to_be_visible()
                assert panel.bounding_box()["width"] == preferred
                page.set_viewport_size({"width": 760, "height": 1000})
                assert panel.bounding_box()["width"] <= 760
                page.set_viewport_size({"width": 390, "height": 844})
                assert panel.bounding_box()["width"] <= 390
                expect(handle).not_to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                page.set_viewport_size({"width": 1440, "height": 1000})
                expect(panel).to_have_css("width", f"{int(preferred)}px")
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
