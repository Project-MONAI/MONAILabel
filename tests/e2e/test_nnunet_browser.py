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

"""CT/MRI nnU-Net setup through the real workspace forms."""

import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e


def test_nnunet_modality_and_automatic_training_settings(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(error.stack))
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.evaluate("""async () => (await fetch('/api/projects', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: 'nnU-Net setup', labels: [
                        {id: 0, name: 'Background'}, {id: 5, name: 'Target'}
                    ]})
                })).json()""")
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                page.locator('#workspace nav [data-page="models"]').click()
                page.locator('[data-action="model-tab"][data-id="training"]').click()
                for modality in ("CT", "MRI"):
                    page.get_by_role("button", name="Create model", exact=True).click()
                    form = page.locator("#action-form")
                    form.get_by_label("Model type", exact=True).select_option("nnunet-v2")
                    form.get_by_label("Model name", exact=True).fill(f"{modality} specialist")
                    expect(form.get_by_label("Scan modality", exact=True)).to_have_value("")
                    form.get_by_label("Scan modality", exact=True).select_option(modality)
                    expect(form).to_contain_text("one consistent MRI sequence")
                    form.get_by_role("button", name="Create model", exact=True).click()
                    card = page.locator("[data-learner-id]").filter(
                        has_text=f"{modality} specialist"
                    )
                    expect(card).to_be_visible()
                    card.get_by_role("button", name="Start training", exact=True).click()
                    form = page.locator("#action-form")
                    form.get_by_text("Training settings (recommended)", exact=True).click()
                    expect(form.get_by_label("Epochs", exact=True)).to_have_value("20")
                    expect(form.get_by_label("Training steps per epoch", exact=True)).to_have_value(
                        "32"
                    )
                    expect(form.get_by_label("Batch size", exact=True)).to_have_count(0)
                    expect(form.locator("[data-training-budget]")).to_have_text(
                        "640 weight updates · nnU-Net plans batch and patch sizes"
                    )
                    page.screenshot(path=str(tmp_path / f"nnunet-{modality}.png"))
                    page.locator("#close-dialog").click()
                learners = page.evaluate(
                    "async (id) => (await fetch(`/api/projects/${id}/learners`)).json()",
                    project["id"],
                )
                assert {learner["config"]["modality"] for learner in learners} == {"CT", "MRI"}
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
