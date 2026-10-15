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

"""Automatic approval, private projects, capacity waiting and administrator directory."""

import os

import httpx
import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e


@pytest.mark.parametrize("gpt", [False, True])
def test_public_signup_and_admin_directory(tmp_path, monkeypatch, gpt):
    from playwright.sync_api import expect, sync_playwright

    for name in (
        "MONAILABEL_DEMO_SELF_REGISTER",
        "MONAILABEL_DEMO_PUBLIC_REGISTER",
        "MONAILABEL_DEMO_CACHE_ONLY",
        "MONAILABEL_DEMO_TRAINING_PREVIEW",
        "MONAILABEL_DEMO_HTTP_LIMITS",
        "MONAILABEL_DEMO_LIFECYCLE",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MONAILABEL_DEMO_CAPACITY", "2")
    monkeypatch.setenv("MONAILABEL_DEMO_GPT", "1" if gpt else "0")
    if gpt:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-browser-fixture-never-sent")
    stack = VideoStack(tmp_path, tmp_path)
    # Metadata-only UI tests never run inference or require real model downloads.
    monkeypatch.setattr("monailabel.server.models.service.prepared", lambda *args: True)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            contexts = [
                browser.new_context(viewport={"width": 1400, "height": 1100}) for _ in range(4)
            ]
            errors = []
            try:
                for index, context in enumerate(contexts[:3]):
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(stack.url)
                    if gpt:
                        expect(page.locator("#demo-notice")).to_contain_text("to OpenAI")
                        if index == 0:
                            page.screenshot(
                                path=str(tmp_path / "gpt-disclosure.png"), full_page=True
                            )
                    page.get_by_label("Username", exact=True).fill(f"signup-{index}")
                    page.get_by_label("Password", exact=True).fill("test-password-1234")
                    page.get_by_label("Confirm password", exact=True).fill("test-password-1234")
                    with page.expect_response("**/api/auth/signup") as response:
                        page.get_by_role("button", name="Create demo account", exact=True).click()
                    assert response.value.status == 201
                    if index < 2:
                        expect(page.locator("#workspace")).to_be_visible()
                        expect(
                            page.get_by_role("heading", name="Your demo workspace is ready.")
                        ).to_be_visible()
                        expect(
                            page.get_by_role("button", name="New project", exact=True)
                        ).to_be_hidden()
                        expect(page.locator("#demo-countdown")).to_contain_text(
                            "Demo time remaining"
                        )
                        projects = context.request.get(stack.url + "/api/projects").json()
                        roles = context.request.get(
                            stack.url + f"/api/projects/{projects[0]['id']}/permissions"
                        ).json()["roles"]
                        assert set(roles) == {"annotator", "reviewer"}
                        models = context.request.get(
                            stack.url + f"/api/projects/{projects[0]['id']}/models"
                        )
                        hosted = [m for m in models.json() if m["provider"] == "openai-polygons"]
                        assert len(hosted) == (2 if gpt else 0)
                        assert "sk-browser-fixture" not in models.text()
                        if gpt and index == 0:
                            page.locator('[data-page="models"]').click()
                            for title in (
                                "GPT-6.1 Sol · demo annotation",
                                "GPT-6 Astra · demo comparison",
                            ):
                                expect(
                                    page.get_by_role("heading", name=title, exact=True)
                                ).to_be_visible()
                            expect(
                                page.get_by_role("button", name="Change provider", exact=True)
                            ).to_have_count(0)
                            page.screenshot(
                                path=str(tmp_path / "visitor-gpt-models.png"), full_page=True
                            )
                            derive = page.get_by_role(
                                "button", name="Create project model", exact=False
                            )
                            expect(derive).to_have_count(3)
                            derive.first.click()
                            page.locator("#action-form input[name=name]").fill(
                                "Visitor model preview"
                            )
                            page.locator("#action-form button[type=submit]").click()
                            expect(page.locator("#dialog")).not_to_be_visible()
                            expect(
                                page.get_by_role(
                                    "heading", name="Visitor model preview", exact=True
                                )
                            ).to_be_visible()
                            expect(
                                page.get_by_role("button", name="Start training", exact=True)
                            ).to_have_count(0)
                            expect(
                                page.get_by_role("button", name="API keys", exact=True)
                            ).to_have_count(0)
                            expect(
                                page.get_by_role("button", name="Create model", exact=True)
                            ).to_be_visible()
                            page.screenshot(
                                path=str(tmp_path / "visitor-model-setup.png"), full_page=True
                            )
                            page.locator('[data-page="overview"]').click()
                        assert context.request.get(stack.url + "/api/auth/accounts").status == 403
                        page.screenshot(
                            path=str(tmp_path / f"approved-visitor-{index}.png"), full_page=True
                        )
                    else:
                        expect(page.locator("#auth-error")).to_contain_text(
                            "Your account is approved"
                        )
                        expect(
                            page.get_by_role("button", name="Sign in", exact=True)
                        ).to_be_visible()
                        expect(page.get_by_label("Password", exact=True)).to_have_value("")
                        expect(page.get_by_label("Confirm password", exact=True)).to_be_hidden()
                        page.screenshot(path=str(tmp_path / "approved-waiting.png"), full_page=True)
                admin = contexts[3].new_page()
                admin.on("pageerror", lambda error: errors.append(str(error)))
                admin.goto(stack.url)
                admin.get_by_role(
                    "button", name="Already have an account? Sign in", exact=True
                ).click()
                admin.get_by_label("Username", exact=True).fill(stack.username)
                admin.get_by_label("Password", exact=True).fill(stack.password)
                admin.get_by_role("button", name="Sign in", exact=True).click()
                expect(admin.locator("#workspace")).to_be_visible()
                admin.locator('[data-page="team"]').click()
                expect(admin.get_by_role("heading", name="Users", exact=True)).to_be_visible()
                directory = admin.locator("#account-directory")
                for index in range(3):
                    expect(
                        directory.get_by_role("cell", name=f"signup-{index}", exact=True)
                    ).to_be_visible()
                expect(
                    admin.get_by_role("cell", name="Approved — not started", exact=True)
                ).to_be_visible()
                admin.screenshot(path=str(tmp_path / "administrator-users.png"), full_page=True)
                assert not errors
            finally:
                for context in contexts:
                    context.close()
                browser.close()
    finally:
        stack.stop_workspace()


def test_prepared_example_buttons_with_real_cache(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    cache = os.environ.get("MONAILABEL_TEST_STARTER_CACHE")
    if not cache:
        pytest.skip("Requires the host's verified public sample cache.")
    monkeypatch.setenv("MONAILABEL_DATASETS_DIR", cache)
    for name in (
        "SELF_REGISTER",
        "PUBLIC_REGISTER",
        "CACHE_ONLY",
        "TRAINING_PREVIEW",
        "HTTP_LIMITS",
        "LIFECYCLE",
    ):
        monkeypatch.setenv("MONAILABEL_DEMO_" + name, "1")
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1440, "height": 1100})
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(stack.url)
            page.get_by_label("Username", exact=True).fill("example-browser-visitor")
            page.get_by_label("Password", exact=True).fill("test-password-1234")
            page.get_by_label("Confirm password", exact=True).fill("test-password-1234")
            page.get_by_role("button", name="Create demo account", exact=True).click()
            expect(page.locator("#workspace")).to_be_visible()
            for heading, button, sample in (
                ("Pathology · QuPath", "Load pathology example", "CMU-1-Small-Region.svs"),
                (
                    "Endoscopy · CVAT",
                    "Load endoscopy video",
                    "HyperKvasir-snare-99b387e7-d07b-4268-9226-4df450c2a198.avi",
                ),
            ):
                page.get_by_text(heading, exact=True).click()
                load = page.get_by_role("button", name=button, exact=True)
                expect(load).to_be_visible(timeout=60000)
                load.click()
                expect(page.get_by_text(sample, exact=True)).to_be_visible(timeout=90000)
            expect(page.get_by_role("button", name="QuPath", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="CVAT", exact=True)).to_be_visible()
            page.screenshot(path=str(tmp_path / "visitor-prepared-examples.png"), full_page=True)
            projects = context.request.get(stack.url + "/api/projects").json()
            pid = projects[0]["id"]
            assert len(context.request.get(f"{stack.url}/api/projects/{pid}/assets").json()) == 2
            assert len(context.request.get(f"{stack.url}/api/projects/{pid}/videos").json()) == 1
            assert (
                context.request.post(
                    stack.url + "/api/demo-access/starter", data={"example": "endoscopy"}
                ).json()
                is None
            )
            assert (
                context.request.post(
                    f"{stack.url}/api/projects/{pid}/dataset-imports",
                    data={"template_id": "Task03_Liver", "limit": 1},
                ).status
                == 403
            )
            assert not errors
            context.close()
            browser.close()
    finally:
        stack.stop_workspace()
