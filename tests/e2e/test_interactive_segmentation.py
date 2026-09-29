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

"""Real viewer controls and direct Update path; deterministic mask inference, no paid models."""

import gzip
import secrets
import threading

import httpx
import nibabel as nib
import numpy as np
import pytest
from conftest import ROOT, VideoStack
from test_browser_desktop import desktop_stack as desktop_stack
from test_browser_desktop import fitted_desktop, observed, settled_input
from test_video_cvat import login_workspace

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord
from monailabel.core.ports import Prediction


def prepare(stack, client, monkeypatch, gate=None):
    service = stack.app.state.services
    service.presets.enabled = True
    project = client.post(
        "/api/projects",
        {
            "name": "Interactive CT",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Spleen"},
                {"id": 2, "name": "Liver"},
            ],
        },
    )
    with service.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=project["id"],
                name="Vision fixture",
                provider="openai-chat-polygons",
                label_ids=[0],
            )
        )
        session.insert(
            ModelRecord(
                project_id=project["id"],
                name="My spleen model",
                mode="from_scratch",
                provider="threshold",
                label_ids=[0, 1],
                config={"thresholds": [0.5]},
            )
        )
    values = np.zeros((32, 36, 20), np.int16)
    values[8:24, 9:27, :] = 100
    response = client.http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "Interactive CT.nii.gz"},
        content=gzip.compress(nib.Nifti1Image(values, np.diag([-1.5, 2, 2.5, 1])).to_bytes()),
    )
    response.raise_for_status()
    calls = []

    class Predictor:
        def predict_prompted(self, image, label, model, spatial, plane, full, progress):
            calls.append((label, spatial, full))
            if gate is not None:
                assert gate.wait(60), "Viewer test did not release the inference worker"
            mask = np.zeros(image.shape[:-1], np.uint8)
            mask[8 if label == 1 else 18 : 14 if label == 1 else 24, 9:14, :] = label
            return Prediction(mask)

    monkeypatch.setattr(service.models, "spatial_provider", lambda model: Predictor())
    automatic_calls = []

    def automatic(project, model, image, prompt, affine):
        automatic_calls.append(model.name)
        mask = np.zeros(image.shape[:-1], np.uint8)
        mask[8:17, 9:20, :] = model.label_ids[-1]
        return mask

    monkeypatch.setattr(service.models, "predict", automatic)
    return project, response.json(), calls, automatic_calls


def queue(stack, name, **arguments):
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id=secrets.token_hex(8), name=name, arguments=arguments)],
        )
    )


@pytest.mark.browser_e2e
def test_ohif_interaction_toolbar(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    artifacts = ROOT / "test-results" / ("ohif-interaction-" + secrets.token_hex(6))
    artifacts.mkdir(parents=True, mode=0o700)
    stack = VideoStack(tmp_path, artifacts)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            client = Client(http=http)
            client.post("/api/auth/setup", {"username": stack.username, "password": stack.password})
            gate = threading.Event()
            project, asset, calls, automatic_calls = prepare(stack, client, monkeypatch, gate)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                context = browser.new_context(viewport={"width": 1600, "height": 1050})
                errors = []
                context.on("page", lambda p: p.on("pageerror", lambda e: errors.append(str(e))))
                try:
                    page = context.new_page()
                    login_workspace(page, stack)
                    page.goto(f"{stack.url}/datasets?project={project['id']}")
                    with page.expect_popup() as opened:
                        page.get_by_role("button", name="OHIF", exact=True).click()
                    viewer = opened.value
                    viewer.wait_for_url("**/ohif/**", timeout=1800000)
                    send = viewer.get_by_role("button", name="Send prompt", exact=True)
                    expect(send).to_be_enabled(timeout=120000)
                    dismiss = viewer.get_by_role("button", name="Confirm and hide")
                    if dismiss.is_visible():
                        dismiss.click()

                    def prompt(text, tool, **args):
                        queue(stack, tool, **args)
                        viewer.get_by_role("textbox", name="Annotation prompt").fill(text)
                        send.click()
                        expect(send).to_be_enabled(timeout=60000)

                    model_selector = viewer.get_by_role("combobox", name="Annotation model")
                    model_selector.select_option(label="nnInteractive")
                    toolbar = viewer.get_by_role("region", name="Interaction toolbar")
                    expect(toolbar).to_be_visible()
                    expect(
                        toolbar.get_by_role("button", name="+ Point", exact=True)
                    ).to_be_disabled()
                    toolbar.get_by_label("Target label", exact=True).fill("Spleen")
                    expect(
                        toolbar.get_by_role("button", name="+ Point", exact=True)
                    ).to_be_enabled()
                    model_selector.select_option(label="VISTA3D")
                    expect(toolbar).to_be_visible()
                    expect(toolbar.get_by_role("button", name="+ Point", exact=True)).to_have_count(
                        0
                    )
                    expect(toolbar.get_by_role("button", name="Update", exact=True)).to_have_text(
                        "Update slice"
                    )
                    assert (
                        model_selector.locator("optgroup").last.get_attribute("label")
                        == "Trained models"
                    )
                    viewer.screenshot(path=str(artifacts / "toolbar-automatic.png"))
                    prompt(
                        "Start nnInteractive for spleen.",
                        "set_interaction_mode",
                        model_name="nnInteractive",
                        target="Spleen",
                    )
                    toolbar = viewer.get_by_role("region", name="Interaction toolbar")
                    expect(toolbar).to_be_visible(timeout=30000)
                    expect(toolbar.get_by_role("toolbar")).to_have_attribute(
                        "title", "nnInteractive · Spleen"
                    )
                    expect(
                        toolbar.get_by_role("button", name="+ Point", exact=True)
                    ).to_have_attribute("aria-pressed", "true")
                    viewer.screenshot(path=str(artifacts / "toolbar-positive.png"))
                    canvas = viewer.locator("canvas").first
                    bounds = canvas.bounding_box()
                    x, y = bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2
                    viewer.mouse.click(x, y)
                    viewer.wait_for_timeout(400)
                    viewer.mouse.wheel(0, 240)
                    viewer.wait_for_timeout(500)
                    viewer.mouse.click(x + 55, y + 45)
                    viewer.wait_for_timeout(400)
                    toolbar.get_by_role("button", name="− Point", exact=True).click()
                    expect(
                        toolbar.get_by_role("button", name="+ Point", exact=True)
                    ).to_have_attribute("aria-pressed", "false")
                    expect(
                        toolbar.get_by_role("button", name="− Point", exact=True)
                    ).to_have_attribute("aria-pressed", "true")
                    viewer.mouse.click(x - 80, y - 55)
                    viewer.wait_for_timeout(400)
                    viewer.mouse.click(x + 80, y + 55)
                    viewer.wait_for_timeout(400)
                    toolbar.get_by_role("button", name="Box", exact=True).click()
                    viewer.mouse.move(x - 35, y - 35)
                    viewer.mouse.down()
                    viewer.mouse.move(x + 35, y + 35, steps=8)
                    viewer.mouse.up()
                    before = len(stack.app.state.services.assistants.provider.calls)
                    toolbar.get_by_role("button", name="Update", exact=True).click()
                    try:
                        expect(
                            toolbar.get_by_role("button", name="Update", exact=True)
                        ).to_be_disabled()
                        expect(
                            toolbar.get_by_role("button", name="+ Point", exact=True)
                        ).to_be_disabled()
                        expect(toolbar.get_by_label("Update options")).to_have_attribute(
                            "aria-disabled", "true"
                        )
                        viewer.screenshot(path=str(artifacts / "toolbar-busy.png"))
                    finally:
                        gate.set()
                    expect(viewer.get_by_role("log")).to_contain_text(
                        "Applied the editable proposal", timeout=60000
                    )
                    assert len(stack.app.state.services.assistants.provider.calls) == before
                    assert len(calls) == 1 and calls[0][2]
                    spatial = calls[0][1]
                    (artifacts / "received-prompts.json").write_text(
                        spatial.model_dump_json(indent=2)
                    )
                    assert spatial.box is not None
                    assert [p.positive for p in spatial.points].count(True) == 2
                    assert [p.positive for p in spatial.points].count(False) == 2
                    assert len({round(p.coordinates[2]) for p in spatial.points}) >= 2
                    viewer.screenshot(path=str(artifacts / "toolbar-mask.png"))
                    viewer.set_viewport_size({"width": 1280, "height": 800})
                    viewer.wait_for_timeout(500)
                    viewer.screenshot(path=str(artifacts / "toolbar-narrow.png"))
                    assert toolbar.evaluate("e => e.scrollWidth <= e.clientWidth")
                    assert viewer.locator(".monailabel-assistant").evaluate(
                        "e => e.scrollWidth <= e.clientWidth"
                    )
                    assert viewer.get_by_role("log").evaluate("e => e.scrollWidth <= e.clientWidth")
                    viewer.set_viewport_size({"width": 1600, "height": 1050})
                    prompt(
                        "Clear negative spleen points in this volume.",
                        "edit_spatial_prompts",
                        operation="clear",
                        kind="point",
                        polarity="negative",
                        target="Spleen",
                        scope="full",
                    )
                    expect(viewer.get_by_role("log")).to_contain_text(
                        "Spatial hints updated", timeout=30000
                    )
                    prompt(
                        "Start nnInteractive for spleen.",
                        "set_interaction_mode",
                        model_name="nnInteractive",
                        target="Spleen",
                    )
                    toolbar.get_by_role("button", name="Update", exact=True).click()
                    expect(
                        viewer.get_by_role("log").get_by_text(
                            "Applied the editable proposal.", exact=False
                        )
                    ).to_have_count(2, timeout=60000)
                    assert len(calls[-1][1].points) == 2 and all(
                        p.positive for p in calls[-1][1].points
                    )
                    toolbar.get_by_label("Target label", exact=True).fill("Liver")
                    toolbar.get_by_role("button", name="+ Point", exact=True).click()
                    viewer.mouse.click(x + 55, y)
                    viewer.wait_for_timeout(400)
                    toolbar.get_by_role("button", name="Update", exact=True).click()
                    expect(
                        viewer.get_by_role("log").get_by_text(
                            "Applied the editable proposal.", exact=False
                        )
                    ).to_have_count(3, timeout=60000)
                    assert calls[-1][0] == 2
                    assert len(calls[-1][1].points) == 1 and calls[-1][1].box is None
                    viewer.screenshot(path=str(artifacts / "toolbar-liver.png"))
                    toolbar.get_by_label("Target label", exact=True).fill("Spleen")
                    toolbar.get_by_role("button", name="Update", exact=True).click()
                    expect(
                        viewer.get_by_role("log").get_by_text(
                            "Applied the editable proposal.", exact=False
                        )
                    ).to_have_count(4, timeout=60000)
                    assert calls[-1][0] == 1 and len(calls[-1][1].points) == 2
                    assert calls[-1][1].box is not None
                    viewer.screenshot(path=str(artifacts / "toolbar-two-labels.png"))
                    toolbar.get_by_label("Update options").click()
                    viewer.screenshot(path=str(artifacts / "toolbar-scope.png"))
                    toolbar.get_by_role("button", name="Update slice", exact=True).click()
                    expect(
                        viewer.get_by_role("log").get_by_text(
                            "Applied the editable proposal.", exact=False
                        )
                    ).to_have_count(5, timeout=60000)
                    assert calls[-1][2] is False
                    expect(toolbar.get_by_role("button", name="Update", exact=True)).to_have_text(
                        "Update slice"
                    )
                    positive = toolbar.get_by_role("button", name="+ Point", exact=True)
                    positive.click()
                    expect(positive).to_have_attribute("aria-pressed", "true")
                    positive.click()
                    expect(positive).to_have_attribute("aria-pressed", "false")
                    positive.click()
                    viewer.keyboard.press("Escape")
                    expect(positive).to_have_attribute("aria-pressed", "false")
                    expect(toolbar).to_be_visible()
                    prompt(
                        "Start nnInteractive for spleen.",
                        "set_interaction_mode",
                        model_name="nnInteractive",
                        target="Spleen",
                    )
                    prompt("Stop interaction mode.", "set_interaction_mode", mode="navigate")
                    expect(positive).to_have_attribute("aria-pressed", "false", timeout=30000)
                    expect(toolbar).to_be_visible()
                    model_selector.select_option(label="VISTA3D")
                    expect(toolbar).to_be_visible()
                    expect(toolbar.get_by_role("button", name="+ Point", exact=True)).to_have_count(
                        0
                    )
                    expect(toolbar.get_by_role("button", name="Update", exact=True)).to_have_text(
                        "Update slice"
                    )
                    assert (
                        model_selector.locator("optgroup").last.get_attribute("label")
                        == "Trained models"
                    )
                    viewer.screenshot(path=str(artifacts / "toolbar-automatic.png"))
                    applied = viewer.get_by_role("log").get_by_text(
                        "Applied the editable proposal.", exact=False
                    )
                    previous = applied.count()
                    toolbar.get_by_role("button", name="Update", exact=True).click()
                    expect(applied).to_have_count(previous + 1, timeout=60000)
                    assert automatic_calls == ["VISTA3D"]
                    prompt(
                        "Clear all inputs in this volume.",
                        "edit_spatial_prompts",
                        operation="clear",
                        kind="all",
                        polarity="all",
                        all_targets=True,
                        scope="full",
                    )
                    assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
                    assert not errors
                finally:
                    for index, window in enumerate(context.pages):
                        if not window.is_closed():
                            window.screenshot(path=str(artifacts / f"page-{index}.png"))
                            (artifacts / f"page-{index}.html").write_text(window.content())
                    context.close()
                    browser.close()
    finally:
        stack.stop_workspace()


@pytest.mark.desktop_e2e
@pytest.mark.parametrize("desktop_stack", [False, True], indirect=True, ids=["http", "https"])
def test_slicer_interaction_toolbar(desktop_stack, monkeypatch):
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    project, asset, calls, automatic_calls = prepare(stack, client, monkeypatch)
    service = stack.app.state.services
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        # The disposable browser accepts this fixture's certificate; both native
        # bridges still verify the real chain and hostname with the supplied CA.
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100}, ignore_https_errors=stack.https
        )
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="Slicer", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            canvas = fitted_desktop(desktop)
            report = (
                service.desktops.runtime.root / desktop.url.rsplit("/", 1)[-1] / "observed.json"
            )
            observed(report, lambda v: v["asset_id"] == asset["id"], desktop)

            def click(point):
                bounds = canvas.bounding_box()
                desktop.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + point[1] * bounds["height"] / int(canvas.get_attribute("height")),
                )
                desktop.wait_for_timeout(150)

            def prompt(text, tool, **args):
                queue(stack, tool, **args)
                value = settled_input(report, desktop)
                click([value["x"], value["y"]])
                desktop.keyboard.type(text)
                desktop.keyboard.press("Control+Enter")

            state = observed(report, lambda v: bool(v["model_names"]), desktop)
            assert state["selected_model"] == "VISTA3D"
            click(state["model_selector"])
            desktop.screenshot(path=str(stack.artifacts / "slicer-model-groups.png"))
            desktop.keyboard.type("nnInteractive")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: v["interaction_visible"], desktop)
            assert state["interaction_mode"] == "navigate"
            # Qt lays out the newly visible target editor after closing its model popup.
            desktop.wait_for_timeout(300)
            state = observed(report, lambda v: v["selected_model"] == "nnInteractive", desktop)
            click(state["target_selector"])
            desktop.keyboard.type("Spleen")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: v["interaction_target"] == "Spleen", desktop)
            click(state["model_selector"])
            desktop.keyboard.type("VISTA3D")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: "+ Point" not in v["hint_buttons"], desktop)
            assert state["interaction_visible"] and not state["hint_separator_visible"]
            assert state["model_groups"][-1] == "Trained models"
            assert state["model_groups"] == [
                "Radiology segmentation",
                "Interactive segmentation",
                "Vision-language models",
                "Trained models",
            ]
            assert not any(state["group_selectable"])
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-automatic.png"))
            prompt(
                "Start nnInteractive for spleen.",
                "set_interaction_mode",
                model_name="nnInteractive",
                target="Spleen",
            )
            state = observed(
                report,
                lambda v: v["interaction_visible"] and v["interaction_mode"] == "positive",
                desktop,
            )
            click(state["view_center"])
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-positive.png"))
            state = observed(report, lambda v: len(v["spatial_objects"]) == 1, desktop)
            desktop.mouse.wheel(0, 120)
            desktop.wait_for_timeout(500)
            click(state["view_center"])
            state = observed(report, lambda v: len(v["spatial_objects"]) == 2, desktop)
            click(state["hint_buttons"]["− Point"])
            state = observed(report, lambda v: v["interaction_mode"] == "negative", desktop)
            click([state["view_center"][0] + 30, state["view_center"][1] + 25])
            state = observed(report, lambda v: len(v["spatial_objects"]) == 3, desktop)
            click(state["hint_buttons"]["Box"])
            state = observed(report, lambda v: v["interaction_mode"] == "box", desktop)
            click([state["view_center"][0] - 30, state["view_center"][1] - 30])
            state = observed(report, lambda v: v["pending_points"] == 1, desktop, timeout=15)
            click([state["view_center"][0] + 30, state["view_center"][1] + 30])
            state = observed(
                report, lambda v: any(i["kind"] == "box" for i in v["spatial_objects"]), desktop
            )
            before = len(service.assistants.provider.calls)
            click(state["hint_buttons"]["Update"])
            state = observed(
                report,
                lambda v: "applied" in v["history"].lower() and 1 in v["mask_labels"],
                desktop,
            )
            assert len(service.assistants.provider.calls) == before
            assert len(calls) == 1 and calls[0][2] and calls[0][1].box
            assert len(calls[0][1].points) == 3
            assert {p.positive for p in calls[0][1].points} == {False, True}
            spleen_voxels = state["mask_counts"]["1"]
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-mask.png"))
            click(state["target_selector"])
            desktop.keyboard.press("Home")
            desktop.keyboard.press("Shift+End")
            desktop.keyboard.press("Backspace")
            desktop.keyboard.type("Liver")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: v["interaction_target"] == "Liver", desktop)
            assert "Spleen" not in state["visible_targets"]
            click(state["hint_buttons"]["+ Point"])
            click(state["view_center"])
            state = observed(report, lambda v: len(v["spatial_objects"]) == 5, desktop)
            click(state["hint_buttons"]["Update"])
            state = observed(report, lambda v: 2 in v["mask_labels"], desktop)
            assert state["mask_counts"]["1"] == spleen_voxels
            assert len(calls) == 2 and calls[-1][0] == 2
            assert len(calls[-1][1].points) == 1 and calls[-1][1].box is None
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-liver.png"))
            click(state["target_selector"])
            desktop.keyboard.press("Home")
            desktop.keyboard.press("Shift+End")
            desktop.keyboard.press("Backspace")
            desktop.keyboard.type("Spleen")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: v["interaction_target"] == "Spleen", desktop)
            assert "Spleen" in state["visible_targets"] and "Liver" not in state["visible_targets"]
            assert len(state["spatial_objects"]) == 5
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-two-labels.png"))
            assert state["hint_separator_visible"]
            click(state["update_menu"])
            state = observed(report, lambda v: bool(v["scope_actions"]), desktop)
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-scope.png"))
            click(state["scope_actions"]["current_slice"])
            state = observed(
                report,
                lambda v: (
                    v["hint_scope"] == "current_slice"
                    and v["history"].lower().count("proposal applied") >= 3
                ),
                desktop,
            )
            assert len(calls) == 3 and calls[-1][2] is False
            assert state["mask_counts"]["2"] > 0
            click(state["update_menu"])
            state = observed(report, lambda v: bool(v["scope_actions"]), desktop)
            click(state["scope_actions"]["full"])
            state = observed(
                report,
                lambda v: (
                    v["hint_scope"] == "full"
                    and v["history"].lower().count("proposal applied") >= 4
                ),
                desktop,
            )
            assert len(calls) == 4 and calls[-1][2] is True
            # Retain the interactive inputs while running a non-interactive model
            # on each native slice plane. Inputs must not reach VISTA3D.
            retained_inputs = state["spatial_objects"]
            click(state["model_selector"])
            desktop.keyboard.type("VISTA3D")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: "+ Point" not in v["hint_buttons"], desktop)
            for count, (view, axis) in enumerate([("Red", 2), ("Yellow", 0), ("Green", 1)], 5):
                click(state["slice_view_selector"])
                state = observed(report, lambda v: bool(v["slice_view_actions"]), desktop)
                desktop.screenshot(
                    path=str(stack.artifacts / ("slicer-view-menu-" + view + ".png"))
                )
                click(state["slice_view_actions"][view])
                state = observed(report, lambda v, view=view: v["slice_view"] == view, desktop)
                assert state["slice"]["axis"] == axis
                click(state["hint_buttons"]["Update"])
                state = observed(
                    report,
                    lambda v, count=count: v["history"].lower().count("proposal applied") >= count,
                    desktop,
                )
                assert state["spatial_objects"] == retained_inputs
            assert automatic_calls == ["VISTA3D"] * 3
            assert len(calls) == 4
            click(state["slice_view_selector"])
            state = observed(report, lambda v: bool(v["slice_view_actions"]), desktop)
            click(state["slice_view_actions"]["Red"])
            prompt(
                "Start nnInteractive for spleen.",
                "set_interaction_mode",
                model_name="nnInteractive",
                target="Spleen",
            )
            state = observed(report, lambda v: v["interaction_mode"] == "positive", desktop)
            mask_hash = state["mask_sha256"]
            click([state["view_center"][0] + 175, state["view_center"][1]])
            state = observed(report, lambda v: len(v["spatial_objects"]) == 6, desktop)
            click(state["hint_buttons"]["Update"])
            state = observed(
                report,
                lambda v: "Move or clear hints outside" in v["history"],
                desktop,
            )
            assert state["mask_sha256"] == mask_hash
            assert len(calls) == 4
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-outside.png"))
            prompt(
                "Clear all inputs in this volume.",
                "edit_spatial_prompts",
                operation="clear",
                kind="all",
                polarity="all",
                all_targets=True,
                scope="full",
            )
            state = observed(report, lambda v: not v["spatial_objects"], desktop)
            assert state["mask_sha256"] == mask_hash
            click(state["model_selector"])
            desktop.keyboard.type("VISTA3D")
            desktop.keyboard.press("Enter")
            state = observed(report, lambda v: "+ Point" not in v["hint_buttons"], desktop)
            assert state["interaction_visible"] and not state["hint_separator_visible"]
            assert state["model_groups"][-1] == "Trained models"
            assert not any(state["group_selectable"])
            desktop.screenshot(path=str(stack.artifacts / "slicer-interaction-automatic.png"))
            prompt(
                "Start MedSAM2 for spleen.",
                "set_interaction_mode",
                model_name="MedSAM2",
                target="Spleen",
            )
            state = observed(
                report,
                lambda v: (
                    v["interaction_visible"]
                    and "+ Point" in v["hint_buttons"]
                    and v["interaction_mode"] == "positive"
                ),
                desktop,
            )
            click(state["hint_buttons"]["+ Point"])
            state = observed(report, lambda v: v["interaction_mode"] == "navigate", desktop)
            click(state["hint_buttons"]["+ Point"])
            observed(report, lambda v: v["interaction_mode"] == "positive", desktop)
            desktop.keyboard.press("Escape")
            observed(report, lambda v: v["interaction_mode"] == "navigate", desktop)
            prompt("Stop interaction mode.", "set_interaction_mode", mode="navigate")
            observed(
                report,
                lambda v: v["interaction_visible"] and v["interaction_mode"] == "navigate",
                desktop,
            )
        finally:
            for index, window in enumerate(context.pages):
                if not window.is_closed():
                    window.screenshot(path=str(stack.artifacts / f"interaction-page-{index}.png"))
            context.close()
            browser.close()


@pytest.mark.desktop_e2e
@pytest.mark.parametrize("desktop_stack", [False, True], indirect=True, ids=["http", "https"])
def test_qupath_interaction_toolbar(desktop_stack, monkeypatch):
    import io
    from types import SimpleNamespace

    from PIL import Image
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    service = stack.app.state.services
    service.presets.enabled = True
    project = client.post("/api/projects", {"name": "Interactive pathology"})
    with service.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=project["id"],
                name="Vision fixture",
                provider="openai-chat-polygons",
                label_ids=[0],
            )
        )
        session.insert(
            ModelRecord(
                project_id=project["id"],
                name="My nuclei model",
                provider="threshold",
                mode="from_scratch",
                label_ids=[0, 1],
                config={"thresholds": [0.5]},
            )
        )
    source = io.BytesIO()
    Image.new("RGB", (256, 256), (220, 170, 190)).save(source, format="PNG")
    response = client.http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "Cells.png"},
        content=source.getvalue(),
    )
    response.raise_for_status()
    asset = response.json()
    calls = []

    def predict(image, label, model, spatial, plane, full, progress):
        calls.append(spatial)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:150, 100:150] = label
        return Prediction(mask)

    monkeypatch.setattr(
        service.models, "spatial_provider", lambda model: SimpleNamespace(predict_prompted=predict)
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100}, ignore_https_errors=stack.https
        )
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="QuPath", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            canvas = fitted_desktop(desktop)
            report = (
                service.desktops.runtime.root / desktop.url.rsplit("/", 1)[-1] / "observed.json"
            )
            value = observed(report, lambda v: v["asset_id"] == asset["id"], desktop)

            def click(point):
                bounds = canvas.bounding_box()
                desktop.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + point[1] * bounds["height"] / int(canvas.get_attribute("height")),
                )
                desktop.wait_for_timeout(150)

            queue(
                stack,
                "set_interaction_mode",
                model_name="SAM 2.1",
                target="Nuclei",
                mode="positive",
            )
            value = settled_input(report, desktop)
            click([value["x"], value["y"]])
            desktop.keyboard.type("Start SAM 2.1 for nuclei.")
            observed(
                report, lambda v: v["draft"] == "Start SAM 2.1 for nuclei.", desktop, timeout=15
            )
            desktop.keyboard.press("Control+Enter")
            value = observed(report, lambda v: v["input_mode"] == "positive", desktop)
            assert "VISTA3D" not in value["models"] and "SAM 2.1" in value["models"]
            click(value["image_center"])
            value = observed(report, lambda v: len(v["inputs"]) == 1, desktop)
            assert len(value["inputs"][0]["coordinates"][0]) == 2
            desktop.screenshot(path=stack.artifacts / "qupath-toolbar-inputs.png")
            click(value["update"])
            value = observed(report, lambda v: v["objects"] > 0, desktop)
            assert len(calls) == 1 and len(calls[0].points) == 1
            desktop.screenshot(path=stack.artifacts / "qupath-toolbar-mask.png")
            assert value["model_groups"] == [
                "Interactive segmentation",
                "Vision-language models",
                "Trained models",
            ]
            click(value["model_selector"])
            desktop.screenshot(path=stack.artifacts / "qupath-model-groups.png")
            desktop.keyboard.press("End")
            desktop.keyboard.press("Enter")
            value = observed(report, lambda v: v["selected_model"] == "My nuclei model", desktop)
            click(value["model_selector"])
            desktop.keyboard.press("ArrowUp")
            desktop.keyboard.press("Enter")
            value = observed(report, lambda v: v["selected_model"] == "Vision fixture", desktop)
            click(value["model_selector"])
            desktop.keyboard.press("ArrowUp")
            desktop.keyboard.press("Enter")
            value = observed(report, lambda v: v["selected_model"] == "SAM 2.1", desktop)
            assert len(value["inputs"]) == 1
            click(value["input_buttons"]["positive"])
            observed(report, lambda v: v["input_mode"] == "positive", desktop)
            desktop.keyboard.press("Escape")
            value = observed(report, lambda v: v["input_mode"] == "navigate", desktop)
            click(value["model_selector"])
            desktop.keyboard.press("End")
            desktop.keyboard.press("Enter")
            value = observed(report, lambda v: v["selected_model"] == "My nuclei model", desktop)
            click(value["update_options"])
            value = observed(report, lambda v: "Update image" in v["menu_actions"], desktop)
            desktop.screenshot(path=stack.artifacts / "qupath-update-menu.png")
            click(value["menu_actions"]["Update image"])
            value = observed(
                report,
                lambda v: v["history"].count("Added editable annotation objects") == 2,
                desktop,
            )
            assert len(value["inputs"]) == 1 and len(calls) == 1
            queue(
                stack,
                "edit_spatial_prompts",
                operation="clear",
                all_targets=True,
                kind="all",
                polarity="all",
                scope="full",
            )
            click([value["x"], value["y"]])
            desktop.keyboard.type("Clear all input points and boxes.")
            desktop.keyboard.press("Control+Enter")
            value = observed(report, lambda v: not v["inputs"], desktop)
            assert value["objects"] > 0
        finally:
            context.close()
            browser.close()
