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

"""Workspace navigation, bundled resources and administrator role assignment."""

import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e


def test_pages_links_model_notices_and_manual_accounts(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    stack.app.state.services.presets.enabled = True
    stack.app.state.services.presets.hosted = {}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1000})
                errors, failures = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "response",
                    lambda response: (
                        failures.append(response.url) if response.status >= 500 else None
                    ),
                )
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                response = page.request.post(
                    stack.url + "/api/projects", data={"name": "Navigation"}
                )
                assert response.status == 201
                project = response.json()
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                for name in ("overview", "datasets", "models", "review", "activity", "team"):
                    page.locator(f'#workspace nav [data-page="{name}"]').click()
                    expect(page.locator("#content")).not_to_be_empty()
                    assert "Select a project" not in page.locator("#content").inner_text()

                page.get_by_role("button", name="Create user", exact=True).click()
                form = page.locator("#action-form")
                form.get_by_label("Username", exact=True).fill("manual-reviewer")
                form.get_by_label("Password (at least 12 characters)", exact=True).fill(
                    stack.password
                )
                form.get_by_role("button", name="Save", exact=True).click()
                expect(page.locator("#dialog")).not_to_be_visible()
                page.get_by_role("button", name="Assign roles", exact=True).click()
                form.get_by_label("User", exact=True).select_option(label="manual-reviewer")
                form.get_by_label("Annotator", exact=True).check()
                form.get_by_label("Reviewer", exact=True).check()
                form.get_by_role("button", name="Save", exact=True).click()
                expect(page.locator("#dialog")).not_to_be_visible()
                row = page.locator("#content tr").filter(has_text="manual-reviewer")
                expect(row).to_contain_text("Annotator")
                expect(row).to_contain_text("Reviewer")
                members = page.request.get(
                    stack.url + f"/api/projects/{project['id']}/members"
                ).json()
                assert next(
                    member["roles"] for member in members if member["username"] == "manual-reviewer"
                ) == ["annotator", "reviewer"]

                page.locator('#workspace nav [data-page="models"]').click()
                cards = page.locator("[data-model-id]")
                expect(cards).to_have_count(6)
                widths = cards.evaluate_all(
                    "cards => cards.map(card => card.getBoundingClientRect().width)"
                )
                assert max(widths) - min(widths) < 2
                page.screenshot(path=str(tmp_path / "model-library.png"), full_page=True)
                for model, text in (
                    ("VISTA3D", "noncommercial research/evaluation"),
                    ("MedSAM2", "research and education"),
                ):
                    card = cards.filter(has=page.get_by_role("heading", name=model, exact=True))
                    card.locator("summary").click()
                    expect(card).to_contain_text(text)
                    assert card.locator('a[href^="https://huggingface.co/"]').count() >= 1
                interactive = cards.filter(
                    has=page.get_by_role("heading", name="nnInteractive", exact=True)
                )
                interactive.locator("summary").click()
                expect(interactive).to_contain_text("CC BY-NC-SA 4.0")
                expect(interactive).to_contain_text("Noncommercial")
                assert (
                    interactive.locator(
                        'a[href="https://creativecommons.org/licenses/by-nc-sa/4.0/"]'
                    ).count()
                    == 1
                )
                for href in page.locator('a[href^="/"]').evaluate_all(
                    "links => [...new Set(links.map(link => link.getAttribute('href')))]"
                ):
                    assert page.request.get(stack.url + href).ok, href
                schema = page.request.get(stack.url + "/openapi.json")
                assert schema.ok
                identifiers = [
                    operation["operationId"]
                    for methods in schema.json()["paths"].values()
                    for operation in methods.values()
                    if isinstance(operation, dict) and "operationId" in operation
                ]
                assert len(identifiers) == len(set(identifiers))
                assert not errors and not failures
                page.screenshot(path=str(tmp_path / "model-notices.png"))
            finally:
                browser.close()
    finally:
        stack.stop_workspace()


def test_ready_viewer_opens_when_workspace_refresh_fails(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    from monailabel.core.models import Asset

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(stack.url + "/?desktop=browser")
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.request.post(
                    stack.url + "/api/projects", data={"name": "Viewer recovery"}
                ).json()
                asset = Asset(
                    project_id=project["id"],
                    name="sample.tif",
                    kind="image2d",
                    split="pool",
                    group_id="synthetic",
                    spatial_shape=[16, 16],
                    image_key="fixture",
                )
                with stack.app.state.services.store.transaction() as session:
                    session.insert(asset)
                page.reload()
                page.locator("#project-select").select_option(project["id"])
                page.locator('#workspace nav [data-page="datasets"]').click()
                expect(page.get_by_role("button", name="QuPath", exact=True)).to_be_visible()
                page.route(
                    "**/api/assets/*/viewer?*", lambda route: route.fulfill(json={"id": "ready"})
                )

                def completed(route):
                    page.route(
                        "**/api/projects/*/assets", lambda request: request.abort("connectionreset")
                    )
                    route.fulfill(
                        json={
                            "id": "ready",
                            "kind": "viewer",
                            "status": "succeeded",
                            "progress": 1,
                            "result": {
                                "viewer": "qupath",
                                "url": "/static/viewer-launch.html?ready",
                            },
                        }
                    )

                page.route("**/api/jobs/ready", completed)
                with page.expect_popup() as opened:
                    page.get_by_role("button", name="QuPath", exact=True).click()
                opened.value.wait_for_url("**/viewer-launch.html?ready", timeout=10000)
                expect(page.locator("#messages")).to_contain_text("Cannot reach the server")
            finally:
                browser.close()
    finally:
        stack.stop_workspace()


def test_export_selected_samples(tmp_path):
    import io
    import json
    import zipfile

    from PIL import Image
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1000})
                page.goto(stack.url)
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                project = page.request.post(
                    stack.url + "/api/projects", data={"name": "Export samples"}
                ).json()
                stream = io.BytesIO()
                Image.new("RGB", (32, 24), (100, 120, 140)).save(stream, format="PNG")
                response = page.request.post(
                    stack.url + f"/api/projects/{project['id']}/assets/upload?name=sample.png",
                    data=stream.getvalue(),
                    headers={"Content-Type": "application/octet-stream"},
                )
                assert response.status == 201
                asset = response.json()
                page.goto(f"{stack.url}/datasets?project={project['id']}")
                expect(page.locator("#dataset-filter")).to_have_value("shared")
                page.locator(f'[data-file-selection="{asset["id"]}"]').check()
                page.get_by_role("button", name="Export selected", exact=True).click()
                form = page.locator("#action-form")
                form.get_by_label("Annotation versions").select_option("all")
                form.get_by_role("button", name="Prepare ZIP", exact=True).click()
                link = page.get_by_role("link", name="Download ZIP", exact=True)
                expect(link).to_be_visible(timeout=30000)
                page.screenshot(path=str(tmp_path / "dataset-export.png"))
                with page.expect_download() as result:
                    link.click()
                download = result.value
                assert download.suggested_filename == "Export_samples-dataset.zip"
                with zipfile.ZipFile(download.path()) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    assert manifest["annotation_versions"] == "all"
                    assert len(manifest["assets"]) == 1
                    assert archive.read(manifest["assets"][0]["image"]) == stream.getvalue()
                form.get_by_role("button", name="Close", exact=True).click()
                page.locator('#workspace nav [data-page="activity"]').click()
                page.get_by_role("button", name="Details", exact=True).click()
                expect(page.get_by_role("link", name="Download ZIP", exact=True)).to_be_visible()
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
