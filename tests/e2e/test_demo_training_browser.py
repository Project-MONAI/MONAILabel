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

"""A recorded training walkthrough never creates a live training job."""

import json
from datetime import UTC, datetime

import httpx
import pytest
from conftest import VideoStack

from monailabel.core.models import Job, Snapshot
from monailabel.server.demo_training import RecordedTraining

pytestmark = pytest.mark.browser_e2e


@pytest.mark.parametrize("vista_available", [False, True])
def test_training_preview_and_recorded_logs_never_start_a_job(
    tmp_path, monkeypatch, vista_available
):
    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_PREVIEW", "1")
    monkeypatch.setenv("MONAILABEL_DEMO_TRAINING_EXAMPLES", str(tmp_path))
    record = RecordedTraining(
        recipe="nnunet-v2",
        name="Recorded test fixture",
        dataset="Synthetic test fixture",
        completed_at=datetime.now(UTC),
        duration_seconds=2,
        training_cases=2,
        evaluation_cases=0,
        epochs=1,
        steps_per_epoch=2,
        checkpoint_sha256="a" * 64,
        source_revision="b" * 40,
        caveats=["Test fixture only."],
        log="fixture checkpoint saved\n<script>bad()</script>",
    )
    (tmp_path / "nnunet-v2.json").write_text(record.model_dump_json())
    vista = record.model_copy(update={"recipe": "vista3d", "name": "VISTA3D fixture"})
    if vista_available:
        (tmp_path / "vista3d.json").write_text(vista.model_dump_json())
    stack = VideoStack(tmp_path / "workspace", tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url) as http:
            http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            ).raise_for_status()
            response = http.post(
                "/api/projects",
                json={
                    "name": "Preview test",
                    "labels": [
                        {"id": 0, "name": "Background"},
                        {"id": 1, "name": "Spleen"},
                    ],
                },
            )
            response.raise_for_status()
            project = response.json()
            http.post(
                f"/api/projects/{project['id']}/learners",
                json={
                    "name": "My model",
                    "recipe": "monai-unet",
                    "label_ids": [0, 1],
                },
            ).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                errors, writes = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "request",
                    lambda req: (
                        writes.append(req.url)
                        if req.method == "POST" and req.url.endswith("/train")
                        else None
                    ),
                )
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                page.locator('nav [data-page="models"]').click()
                page.get_by_role("button", name="Training", exact=True).click()
                cards = page.locator("[data-recorded-recipe]")
                expect(cards).to_have_count(2)
                nnunet = cards.nth(0)
                vista_card = cards.nth(1)
                expect(nnunet.get_by_role("heading", name="nnU-Net training")).to_be_visible()
                expect(
                    vista_card.get_by_role("heading", name="VISTA3D fine-tuning")
                ).to_be_visible()
                expect(nnunet).to_contain_text("No held-out score is available")
                expect(page.locator("#dialog")).not_to_be_visible()
                if not vista_available:
                    expect(vista_card).to_contain_text("not prepared yet")
                    (tmp_path / "vista3d.json").write_text(vista.model_dump_json())
                    vista_card.get_by_role("button", name="Retry loading example").click()
                expect(vista_card).to_contain_text("Recorded run — completed before your visit")
                nnunet.locator("summary").click()
                expect(nnunet.get_by_label("Recorded training log")).to_have_value(record.log)
                expect(nnunet.get_by_label("Recorded training log")).to_be_visible()
                assert nnunet.locator("script").count() == 0
                page.wait_for_timeout(3500)
                expect(nnunet.locator("details")).to_have_attribute("open", "")
                page.screenshot(path=str(tmp_path / "training-cards.png"), full_page=True)
                page.set_viewport_size({"width": 834, "height": 1194})
                page.get_by_role("button", name="Hide assistant", exact=True).click()
                expect(nnunet).to_be_visible()
                expect(vista_card).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(tmp_path / "training-cards-tablet.png"), full_page=True)
                page.set_viewport_size({"width": 1440, "height": 1100})
                nnunet.locator("summary").click()
                page.get_by_role("button", name="Preview training", exact=True).click()
                host = page.locator("[data-training-preview]")
                expect(host).to_contain_text("no training will run")
                expect(host).to_contain_text("does not validate training eligibility")
                host.get_by_label("Recorded example", exact=True).select_option("nnunet-v2")
                expect(host).to_contain_text("Recorded run — completed before your visit")
                expect(host).to_contain_text("No held-out score is available")
                expect(host.get_by_label("Recorded training log")).to_have_value(record.log)
                page.screenshot(path=str(tmp_path / "training-preview.png"), full_page=True)
                host.get_by_label("Recorded example", exact=True).select_option("vista3d")
                expect(host.get_by_label("Recorded training log")).to_have_value(vista.log)
                assert not errors and not writes
                assert stack.app.state.services.store.list(Job) == []
                assert stack.app.state.services.store.list(Snapshot) == []
                (tmp_path / "browser-check.json").write_text(
                    json.dumps({"jobs_started": 0, "passed": True})
                )
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
