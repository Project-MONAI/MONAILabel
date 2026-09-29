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

"""Dataset choices expose labels, grouping and license terms in a real browser."""

import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e


def test_template_terms_and_import_choices(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.evaluate("""async () => (await fetch('/api/projects', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: 'Template choices'})
                })).json()""")
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                page.locator('#workspace nav [data-page="datasets"]').click()
                page.get_by_role("button", name="Sample datasets", exact=True).click()
                form = page.locator("#action-form")
                form.locator('[name="template_id"]').select_option("tnbc-nuclei")
                expect(form).to_contain_text("CC BY 4.0")
                expect(form).to_contain_text("Naylor")
                expect(form).to_contain_text("11 published patient groups")
                expect(form.get_by_role("link", name="Labels and structures")).to_have_attribute(
                    "href", "https://zenodo.org/records/1175282"
                )
                form.locator('[name="split"]').select_option("validation")
                expect(form.locator('[name="include_masks"]')).to_have_value("yes")
                expect(form.get_by_role("button", name="Import samples")).to_be_enabled()
                expect(form.locator("#dataset-mask-summary")).to_contain_text(
                    "accepted on import and ready for evaluation"
                )
                form.locator('[name="split"]').select_option("pool")
                expect(form.locator("#dataset-mask-summary")).to_contain_text(
                    "Review them before training"
                )
                form.locator('[name="split"]').select_option("validation")
                form.locator('[name="template_id"]').select_option("kvasir-instrument")
                expect(form).to_contain_text("commercial use requires prior written permission")
                expect(form).to_contain_text("no patient/procedure mapping")
                expect(form.get_by_role("link", name="Labels and structures")).to_have_attribute(
                    "href", "https://datasets.simula.no/kvasir-instrument/"
                )
                expect(form.get_by_role("link", name="Usage terms")).to_have_attribute(
                    "href", "https://datasets.simula.no/kvasir-instrument/#terms-of-use"
                )
                expect(form.locator('[name="include_masks"]')).to_have_value("yes")
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()


def test_uploaded_evaluation_labels_are_ready(tmp_path):
    import numpy as np
    from PIL import Image
    from playwright.sync_api import expect, sync_playwright

    image, mask = tmp_path / "case.png", tmp_path / "case_mask.png"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(image)
    values = np.zeros((8, 8), dtype=np.uint8)
    values[2:6, 2:6] = 1
    Image.fromarray(values).save(mask)
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.evaluate("""async () => (await fetch('/api/projects', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: 'Evaluation references'})
                })).json()""")
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                page.locator('#workspace nav [data-page="datasets"]').click()
                page.get_by_role("button", name="Import files", exact=True).click()
                form = page.locator("#action-form")
                form.locator('[name="content"]').select_option("labels")
                expect(form.locator('[name="reviewed"]')).to_be_visible()
                form.locator('[name="split"]').select_option("validation")
                expect(form.locator('[name="reviewed"]')).to_be_hidden()
                form.locator('[name="images"]').set_input_files(image)
                form.locator('[name="labels"]').set_input_files(mask)
                form.locator("[data-label-name]").fill("Structure")
                form.locator('[name="set_name"]').fill("Ready references")
                form.get_by_role("button", name="Import files", exact=True).click()
                expect(page.locator("#dialog")).not_to_be_visible()
                results = page.evaluate(
                    """async (id) => Promise.all(['decisions', 'evaluation-set-versions'].map(
                        async path => (await fetch(`/api/projects/${id}/${path}`)).json()))""",
                    project["id"],
                )
                assert len(results[0]) == 1 and results[0][0]["verdict"] == "accepted"
                assert len(results[1]) == 1 and len(results[1][0]["samples"]) == 1
            finally:
                browser.close()
    finally:
        stack.stop_workspace()


def test_totalsegmentator_models_and_training_setup(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    from monailabel.providers.catalog.presets import PresetConnection

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    stack.app.state.services.presets.enabled = True
    stack.app.state.services.presets.hosted = {
        "nvidia-astra": PresetConnection(
            "openai-polygons", {"url": "https://unused.test", "model": "gpt-6-astra"}
        ),
        "nvidia-claude-opus-5": PresetConnection(
            "anthropic-polygons", {"url": "https://unused.test", "model": "claude-opus-5"}
        ),
        "gemini-flash": PresetConnection(
            "openai-chat-polygons", {"url": "https://unused.test", "model": "gemini-3.8-flash"}
        ),
    }
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.evaluate("""async () => (await fetch('/api/projects', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: 'CT and MRI models'})
                })).json()""")
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                page.locator('#workspace nav [data-page="models"]').click()
                for group, count in [("radiology", 3), ("interactive", 2), ("vision", 3)]:
                    expect(
                        page.locator(f'.model-group[aria-labelledby="model-group-{group}"] article')
                    ).to_have_count(count)
                cards = page.locator("article[data-model-id]")
                expect(cards.locator(".model-symbol svg")).to_have_count(7)
                logo = cards.locator(".model-symbol img")
                expect(logo).to_have_count(1)
                assert logo.evaluate(
                    "async image => { await image.decode(); return image.naturalWidth > 0; }"
                )
                expect(cards.filter(has_text="GPT-6 Astra")).to_contain_text("Default")
                for width in (1600, 1100, 390):
                    page.set_viewport_size({"width": width, "height": 1100})
                    if width == 390:
                        page.locator("#close-assistant").click()
                    sizes = cards.evaluate_all(
                        "cards => cards.map(card => ({w: card.offsetWidth, h: card.offsetHeight}))"
                    )
                    assert max(s["w"] for s in sizes) - min(s["w"] for s in sizes) <= 1
                    assert max(s["h"] for s in sizes) - min(s["h"] for s in sizes) <= 1
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    page.screenshot(path=tmp_path / f"model-cards-{width}.png", full_page=True)
                page.set_viewport_size({"width": 1600, "height": 1100})
                for name, count in [("TotalSegmentator CT", 117), ("TotalSegmentator MRI", 50)]:
                    card = page.locator("article[data-model-id]").filter(
                        has=page.get_by_role("heading", name=name, exact=True)
                    )
                    expect(card).to_contain_text(f"{count} structures")
                    expect(card).to_contain_text("3 mm")
                    card.locator("summary").click()
                    expect(card).to_contain_text("Apache-2.0")
                    card.get_by_role("button", name=f"Supported structures · {count}").click()
                    page.locator("#target-search").fill("liver")
                    expect(page.locator("#target-results")).to_contain_text("Liver")
                    page.get_by_role("button", name="Done", exact=True).click()
                card.get_by_role("button", name="Create project model").click()
                form = page.locator("#action-form")
                form.get_by_label("Model name", exact=True).fill("MRI specialist")
                form.get_by_role("button", name="Create model", exact=True).click()
                expect(
                    page.get_by_role("heading", name="MRI specialist", exact=True)
                ).to_be_visible()
                learner = page.evaluate(
                    "async (id) => (await fetch(`/api/projects/${id}/learners`)).json()",
                    project["id"],
                )[0]
                assert learner["recipe"] == "totalsegmentator-mr" and learner["inherit_targets"]
                page.locator('#workspace nav [data-page="datasets"]').click()
                page.get_by_role("button", name="Sample datasets", exact=True).click()
                form = page.locator("#action-form")
                form.locator('[name="template_id"]').select_option("totalsegmentator-mr")
                expect(form).to_contain_text("CC BY 4.0")
                expect(form).to_contain_text("5.1 GB")
                expect(form.get_by_role("link", name="Labels and structures")).to_have_attribute(
                    "href", "https://github.com/wasserth/TotalSegmentator#class-details"
                )
                form.locator('[name="include_masks"]').select_option("yes")
                form.get_by_text("Advanced options", exact=True).click()
                expect(form.locator('[name="targets"][value="liver"]')).to_be_visible()
                assert not errors
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
