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

"""Prepared public samples in real viewers, without elevating visitor roles."""

import os

import pytest
from test_browser_desktop import desktop_stack as desktop_stack
from test_browser_desktop import fitted_desktop, observed, settled_input
from test_cvat_managed import video_stack as video_stack
from test_video_cvat import draw_rectangle, open_editor

from monailabel.client.client import Client


@pytest.fixture(autouse=True)
def prepared_configuration(isolated_presets, monkeypatch):
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


def visitor_example(stack, client, example):
    credentials = {"username": "prepared-" + example, "password": stack.password}
    client.http.cookies.clear()
    client.post("/api/auth/signup", credentials)
    client.post("/api/auth/login", credentials)
    pid = client.get("/api/projects")[0]["id"]
    for job in client.get(f"/api/projects/{pid}/jobs"):
        client.wait(job["id"], timeout=120)
    job = client.post("/api/demo-access/starter", {"example": example})
    client.wait(job["id"], timeout=180)
    assert set(client.get(f"/api/projects/{pid}/permissions")["roles"]) == {
        "annotator",
        "reviewer",
    }
    assert client.http.get("/api/auth/accounts").status_code == 403
    return pid, credentials


def sign_in(page, stack, credentials, pid):
    from playwright.sync_api import expect

    page.goto(stack.url)
    page.get_by_role("button", name="Already have an account? Sign in", exact=True).click()
    page.get_by_label("Username", exact=True).fill(credentials["username"])
    page.get_by_label("Password", exact=True).fill(credentials["password"])
    page.get_by_role("button", name="Sign in", exact=True).click()
    expect(page.locator("#workspace")).to_be_visible()
    page.goto(f"{stack.url}/datasets?project={pid}")


@pytest.mark.desktop_e2e
def test_prepared_qupath_for_public_visitor(desktop_stack, monkeypatch):
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    pid, credentials = visitor_example(stack, client, "pathology")
    slide = next(a for a in client.get(f"/api/projects/{pid}/assets") if a["kind"] == "image2d")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        try:
            page = context.new_page()
            sign_in(page, stack, credentials, pid)
            with page.expect_popup() as opened:
                page.get_by_role("button", name="QuPath", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=120000)
            canvas = fitted_desktop(desktop)
            runtime = stack.app.state.services.desktops.runtime
            report = runtime.root / desktop.url.rsplit("/", 1)[-1] / "observed.json"
            value = observed(report, lambda v: v["asset_id"] == slide["id"], desktop)
            assert "SAM 2.1" in value["models"]
            assert "VISTA3D" not in value["models"]
            # Wait for actual slide pixels, not just an allocated VNC framebuffer.
            desktop.wait_for_function(
                """() => {
                    const canvas = document.querySelector('#display').contentDocument
                        .querySelector('#noVNC_container canvas');
                    const pixels = canvas.getContext('2d').getImageData(
                        0, 0, canvas.width, canvas.height).data;
                    let tissue = 0;
                    for (let i = 0; i < pixels.length; i += 64)
                        if (pixels[i] > 120 && pixels[i + 1] > 40 && pixels[i + 2] > 60
                            && pixels[i] > pixels[i + 1] * 1.1) tissue++;
                    return tissue > 500;
                }""",
                timeout=15000,
            )

            def click(point):
                bounds = canvas.bounding_box()
                desktop.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + point[1] * bounds["height"] / int(canvas.get_attribute("height")),
                )

            value = settled_input(report, desktop)
            click(value["target_selector"])
            desktop.keyboard.press("Control+A")
            desktop.keyboard.type("Nucleus")
            desktop.keyboard.press("Enter")
            value = observed(report, lambda v: v["target"] == "Nucleus", desktop)
            click(value["input_buttons"]["positive"])
            observed(report, lambda v: v["input_mode"] == "positive", desktop)
            desktop.screenshot(path=stack.artifacts / "visitor-qupath-public-slide.png")
        finally:
            context.close()
            browser.close()


@pytest.mark.video_e2e
def test_prepared_cvat_for_public_visitor(video_stack, browser_session, video_http, monkeypatch):
    from playwright.sync_api import expect

    stack = video_stack
    # The host prewarms the shared service; the visitor may only reuse it.
    stack.app.state.services.video_editors.configured()
    client = Client(http=video_http)
    pid, credentials = visitor_example(stack, client, "endoscopy")
    video = client.get(f"/api/projects/{pid}/videos")[0]
    context, errors = browser_session
    page = context.new_page()
    sign_in(page, stack, credentials, pid)
    _, viewer = open_editor(page, video_http, video["id"], keep_page=True)
    native = viewer.frame_locator("#editor")
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=60000)
    viewer.wait_for_function("document.querySelector('#editor').contentWindow.monaiVideo?.ready()")
    selector = viewer.locator("#detection-model")
    selector.select_option(label="SAM 2.1")
    viewer.locator("#target-label").fill("Snare")
    toolbar = viewer.get_by_role("toolbar", name="Interaction toolbar")
    expect(toolbar.get_by_role("button", name="+ Point", exact=True)).to_be_visible()
    # A disposable manual draft verifies native editing and save authorization.
    draw_rectangle(viewer.frames[1], "Snare", [100, 100, 200, 200], source_size=(720, 576))
    viewer.locator("#save").click()
    expect(viewer.locator("#status")).to_have_text("CVAT draft saved.")
    viewer.screenshot(path=stack.artifacts / "visitor-cvat-public-video.png")
    assert client.get(f"/api/projects/{pid}/videos")[0]["revision"] == 0
    assert not errors
