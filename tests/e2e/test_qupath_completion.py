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

import io

import pytest
from PIL import Image
from test_browser_desktop import desktop_stack as desktop_stack
from test_browser_desktop import fitted_desktop, observed, settled_input
from test_video_cvat import login_workspace

from monailabel.core.models import ReviewDecision

pytestmark = pytest.mark.desktop_e2e


@pytest.mark.parametrize("review", [False, True])
def test_qupath_astra_hint_submit_continue_and_exit(desktop_stack, monkeypatch, review):
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    monkeypatch.setenv("MONAILABEL_DEMO_GPT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-requests-made")
    project = client.post(
        "/api/projects",
        {
            "name": "Disposable QuPath",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Nucleus"}],
        },
    )
    pid = project["id"]
    source = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 170, 190)).save(source, format="PNG")
    asset = client.http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"group_id": "synthetic-slide", "name": "Synthetic slide.png"},
        content=source.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    ).json()
    mask = [[1 if 20 < x < 40 and 20 < y < 40 else 0 for x in range(64)] for y in range(64)]
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask, "covered_labels": [0, 1]},
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/{'reviews' if review else 'datasets'}?project={pid}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="QuPath", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=180000)
            canvas = fitted_desktop(desktop)
            runtime = stack.app.state.services.desktops.runtime
            report = runtime.root / desktop.url.rsplit("/", 1)[-1] / "observed.json"
            value = observed(report, lambda v: v["asset_id"] == asset["id"], desktop)
            value = settled_input(report, desktop)
            assert value["selected_model"] == "GPT-6 Astra · demo comparison"
            assert (
                value["prompt_hint"] == "Segment all nuclei in the selected region using GPT Astra."
            )

            def click(point):
                bounds = canvas.bounding_box()
                desktop.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + point[1] * bounds["height"] / int(canvas.get_attribute("height")),
                )

            desktop.screenshot(path=stack.artifacts / "qupath-ready.png")
            click(value["bad"] if review else value["submit"])
            value = observed(report, lambda v: "Keep working" in v["dialog"], desktop)
            first_revision = 1 if review else 2
            assert value["revision"] == first_revision
            desktop.screenshot(path=stack.artifacts / "qupath-submitted.png")
            click(value["dialog"]["Keep working"])
            value = observed(report, lambda v: not v["dialog"], desktop)
            assert value["revision"] == first_revision
            click(value["good"] if review else value["submit"])
            value = observed(report, lambda v: "Return to workspace" in v["dialog"], desktop)
            # An unchanged mask is reviewed in place, not duplicated as a new revision.
            final_revision = 1 if review else 3
            assert value["revision"] == final_revision
            if review:
                assert value["status"].startswith("Good · revision 1")
                decisions = stack.app.state.services.store.list(ReviewDecision, pid)
                assert {d.verdict for d in decisions if d.asset_id == asset["id"]} >= {
                    "accepted",
                    "changes_requested",
                }
                desktop.screenshot(path=stack.artifacts / "qupath-reviewed.png")
            with desktop.expect_event("close", timeout=30000):
                click(value["dialog"]["Return to workspace"])
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == final_revision
        finally:
            context.close()
            browser.close()
