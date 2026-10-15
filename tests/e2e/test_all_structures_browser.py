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

"""Real OHIF/Slicer controls with a deterministic all-structure mask, no model download."""

import gzip
import hashlib
import json

import httpx
import nibabel as nib
import numpy as np
import pytest
from conftest import VideoStack
from test_browser_desktop import desktop_stack, fitted_desktop, observed  # noqa: F401
from test_demo_session import account_deadline, countdown, expire_visitor, timed_visitor
from test_video_cvat import login_workspace

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord, Role
from monailabel.core.ports import Prediction
from monailabel.server.storage_quota import exceeded


def prepare(stack, client, monkeypatch):
    project = client.post("/api/projects", {"name": "All structures"})
    prefix = "/api/projects/" + project["id"]
    image = np.zeros((24, 28, 12), np.int16)
    image[5:19, 6:22, 2:10] = 100
    response = client.http.post(
        prefix + "/assets/upload",
        params={"name": "synthetic-ct.nii.gz"},
        content=gzip.compress(nib.Nifti1Image(image, np.eye(4)).to_bytes()),
    )
    response.raise_for_status()
    asset = response.json()
    with stack.app.state.services.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=project["id"],
                name="TotalSegmentator CT",
                provider="totalsegmentator-ct",
                label_ids=[0],
                read_only=True,
            )
        )
    calls = []

    class Segmenter:
        def predict_volume(self, volume, labels, prompt, model):
            ids = [label.id for label in labels if label.id]
            calls.append(ids)
            mask = np.zeros(volume.image.shape[:-1], np.uint8)
            mask[4:17, 8:17, 4 + len(calls)] = np.asarray(ids, np.uint8).reshape(13, 9)
            return Prediction(mask)

    monkeypatch.setattr(
        stack.app.state.services.models.recipes, "segmenter", lambda *args: Segmenter()
    )
    return project, asset, calls


def replacement(stack):
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="replace-all",
                    name="annotate",
                    arguments={
                        "model_name": "TotalSegmentator CT",
                        "all_targets": True,
                        "replace_existing": True,
                        "scope": "full",
                    },
                )
            ],
        )
    )


@pytest.mark.browser_e2e
def test_ohif_all_structures_button_and_submit(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            client = Client(http=http)
            client.post("/api/auth/setup", {"username": stack.username, "password": stack.password})
            project, asset, calls = prepare(stack, client, monkeypatch)
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1600, "height": 1100})
                try:
                    login_workspace(page, stack)
                    page.goto(f"{stack.url}/datasets?project={project['id']}")
                    with page.expect_popup() as opened:
                        page.get_by_role("button", name="OHIF", exact=True).click()
                    viewer = opened.value
                    viewer.wait_for_url("**/ohif/**", timeout=120000)
                    expect(
                        viewer.get_by_role("button", name="Send prompt", exact=True)
                    ).to_be_enabled(timeout=120000)
                    dismiss = viewer.get_by_role("button", name="Confirm and hide")
                    if dismiss.is_visible():
                        dismiss.click()
                    viewer.get_by_role("combobox", name="Annotation model").select_option(
                        label="TotalSegmentator CT"
                    )
                    button = viewer.get_by_role(
                        "button", name="Segment all available labels", exact=True
                    )
                    expect(button).to_contain_text("117")
                    expect(viewer.get_by_label("Target label", exact=True)).to_have_value("")
                    with viewer.expect_response("**/spatial-inference") as response:
                        button.click()
                    assert response.value.status == 200
                    reply = response.value.json()
                    client.wait(reply["job_id"])
                    expect(
                        viewer.get_by_text("Applied the editable proposal.", exact=False)
                    ).to_be_visible(timeout=30000)
                    viewer.locator("canvas").first.hover()
                    for _ in range(5):
                        viewer.mouse.wheel(0, 120)
                        viewer.wait_for_timeout(150)
                    viewer.screenshot(path=str(tmp_path / "ohif-all-117.png"))
                    assert len(calls) == 1 and len(calls[0]) == 117
                    replacement(stack)
                    viewer.get_by_role("textbox", name="Annotation prompt").fill(
                        "delete the current segmentation and run TotalSegmentator CT on all labels"
                    )
                    with viewer.expect_response("**/assistant") as replaced:
                        viewer.get_by_role("button", name="Send prompt", exact=True).click()
                    assert replaced.value.status == 200
                    client.wait(replaced.value.json()["job_id"])
                    expect(
                        viewer.get_by_text("Applied the editable proposal.", exact=False)
                    ).to_have_count(2, timeout=30000)
                    submit = viewer.get_by_role("button", name="Submit for review", exact=True)

                    # Inject only the full-budget decision; the real storage path,
                    # HTTP error and viewer draft/retry behavior stay in use. Exact
                    # byte/object exhaustion is covered by test_storage_quota.py.
                    def reject_save(*args):
                        raise exceeded()

                    with monkeypatch.context() as patch:
                        patch.setattr(
                            stack.app.state.services.artifacts.quota, "check", reject_save
                        )
                        with viewer.expect_response("**/review-mask?*") as refused:
                            submit.click()
                        assert refused.value.status == 507
                        expect(
                            viewer.get_by_text("Demo storage is full.", exact=False)
                        ).to_be_visible()
                        assert client.get(f"/api/assets/{asset['id']}/annotations") == []
                        viewer.screenshot(path=str(tmp_path / "ohif-quota-draft-retained.png"))
                    # Retry the same in-memory mask without inference or reconstruction.
                    with viewer.expect_response("**/review-mask?*") as saved:
                        submit.click()
                    assert saved.value.status == 201
                    assert len(saved.value.json()["covered_labels"]) == 118
                    annotation_id = saved.value.json()["id"]
                    binary = client.http.get(f"/api/annotations/{annotation_id}/mask.bin").content
                    assert set(np.frombuffer(binary, np.uint8)) == set(range(118))
                    result = np.frombuffer(binary, np.uint8).reshape((24, 28, 12))
                    assert not np.any(result[:, :, 5])
                    assert np.count_nonzero(result[:, :, 6]) == 117
                    assert len(calls) == 2
                finally:
                    browser.close()
    finally:
        stack.stop_workspace()


@pytest.mark.browser_e2e
def test_ohif_shared_account_countdown_and_expiry(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            client = Client(http=http)
            client.post("/api/auth/setup", {"username": stack.username, "password": stack.password})
            project, _, _ = prepare(stack, client, monkeypatch)
            identifier = timed_visitor(stack, client)
            stack.app.state.services.auth.add_member(project["id"], identifier, [Role.MANAGER])
            with sync_playwright() as p:
                browser = p.chromium.launch()
                events = []
                try:
                    page = browser.new_page(viewport={"width": 1600, "height": 1100})
                    page.context.on(
                        "page",
                        lambda window: window.on(
                            "pageerror", lambda error: events.append(str(error))
                        ),
                    )
                    page.context.on(
                        "response",
                        lambda response: (
                            events.append(f"{response.status} {response.url.split('?')[0]}")
                            if response.status >= 400
                            else None
                        ),
                    )
                    login_workspace(page, stack)
                    page.goto(f"{stack.url}/datasets?project={project['id']}")
                    deadline = account_deadline(page)
                    with page.expect_popup() as opened:
                        page.get_by_role("button", name="OHIF", exact=True).click()
                    viewer = opened.value
                    viewer.wait_for_url("**/ohif/**", timeout=120000)
                    banner = ".monailabel-demo-session [role=timer]"
                    countdown(viewer, banner)
                    assert account_deadline(viewer) == deadline
                    assert abs(countdown(page) - countdown(viewer, banner)) <= 2
                    dismiss = viewer.get_by_role("button", name="Confirm and hide")
                    if dismiss.is_visible():
                        dismiss.click()
                    viewer.get_by_role("tab", name="Manual editing").click()
                    countdown(viewer, banner)
                    viewer.get_by_role("tab", name="Assistant", exact=True).click()
                    before = countdown(viewer, banner)
                    viewer.wait_for_timeout(2100)
                    assert countdown(viewer, banner) < before
                    viewer.reload()
                    countdown(viewer, banner)
                    assert account_deadline(viewer) == deadline
                    prompt = viewer.get_by_role("textbox", name="Annotation prompt")
                    prompt.fill("Unsaved local test draft")
                    viewer.screenshot(path=str(tmp_path / "ohif-countdown.png"))
                    expire_visitor(stack, identifier)
                    viewer.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                    expect(viewer.locator(banner)).to_have_text("Demo access has ended.")
                    expect(prompt).to_have_value("Unsaved local test draft")
                    assert (
                        viewer.evaluate("async () => (await fetch('/api/projects')).status") == 401
                    )
                    expect(
                        viewer.get_by_role("link", name="Return to the workspace")
                    ).to_be_visible()
                finally:
                    for context in browser.contexts:
                        for index, window in enumerate(context.pages):
                            window.screenshot(path=str(tmp_path / f"ohif-expiry-{index}.png"))
                            (tmp_path / f"ohif-expiry-{index}.html").write_text(window.content())
                    (tmp_path / "ohif-expiry-events.json").write_text(json.dumps(events))
                    browser.close()
    finally:
        stack.stop_workspace()


@pytest.mark.desktop_e2e
def test_slicer_all_structures_button(desktop_stack, monkeypatch):  # noqa: F811
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    project, asset, calls = prepare(stack, client, monkeypatch)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        try:
            page = browser.new_page(viewport={"width": 1800, "height": 1100})
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="Slicer", exact=True).click()
            viewer = opened.value
            viewer.wait_for_url("**/desktop/*", timeout=180000)
            canvas = fitted_desktop(viewer)
            report = (
                stack.app.state.services.desktops.runtime.root
                / viewer.url.rsplit("/", 1)[-1]
                / "observed.json"
            )

            def click(point):
                bounds = canvas.bounding_box()
                viewer.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + point[1] * bounds["height"] / int(canvas.get_attribute("height")),
                )

            value = observed(
                report, lambda v: v["asset_id"] == asset["id"] and bool(v["model_names"]), viewer
            )
            click(value["model_selector"])
            viewer.keyboard.type("TotalSegmentator CT")
            viewer.keyboard.press("Enter")
            observed(report, lambda v: v["all_targets_visible"], viewer)
            viewer.wait_for_timeout(500)
            value = observed(report, lambda v: v["selected_model"] == "TotalSegmentator CT", viewer)
            assert value["interaction_target"] == ""
            click(value["all_targets_button"])
            value = observed(report, lambda v: len(v["mask_labels"]) == 118, viewer, timeout=120)
            assert value["mask_labels"] == list(range(118))
            assert len(calls) == 1 and len(calls[0]) == 117
            viewer.screenshot(path=str(stack.artifacts / "slicer-all-117.png"))
            replacement(stack)
            click([value["x"], value["y"]])
            viewer.keyboard.type(
                "delete the current segmentation and run TotalSegmentator CT on all labels"
            )
            viewer.keyboard.press("Control+Enter")
            expected = np.zeros((24, 28, 12), np.uint8)
            expected[4:17, 8:17, 6] = np.asarray(calls[0], np.uint8).reshape(13, 9)
            digest = hashlib.sha256(expected.tobytes()).hexdigest()
            observed(report, lambda v: v["mask_sha256"] == digest, viewer, timeout=120)
            assert len(calls) == 2
        finally:
            browser.close()
