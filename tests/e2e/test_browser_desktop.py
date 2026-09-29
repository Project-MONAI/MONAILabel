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

"""Real native viewers, input streaming and reconnects from a browser-only client."""

import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import time

import httpx
import nibabel as nib
import numpy as np
import pytest
from conftest import ROOT, VideoStack
from PIL import Image
from test_video_cvat import login_workspace

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord, Proposal, User
from monailabel.core.ports import Prediction
from monailabel.server.dataset_downloads import Downloads
from monailabel.server.desktops.models import DesktopSession
from monailabel.viewers import browser_desktop

pytestmark = pytest.mark.desktop_e2e

SLICER_PROBE = """
# Test observation only; the production bridge above handles all viewer actions.
import hashlib
import json
from pathlib import Path
def observe_desktop():
    dock = slicer.monailabelAssistant
    if not dock.asset or dock.loading or dock.future or dock.job_id:
        return
    point = dock.prompt.mapToGlobal(qt.QPoint(20, 20))
    def center(widget):
        point = widget.mapToGlobal(qt.QPoint(widget.width // 2, widget.height // 2))
        return [point.x(), point.y()]
    modal = qt.QApplication.activeModalWidget()
    buttons = (
        {str(button.text): center(button) for button in modal.buttons()}
        if isinstance(modal, qt.QMessageBox) else {}
    )
    Path('/session/observed.json').write_text(json.dumps({
        'asset_id': dock.asset['id'], 'revision': dock.asset['revision'],
        'mode': 'review' if dock.review_mode else 'annotation',
        'shared_filesystem': dock.shared_filesystem,
        'mask_labels': [int(x) for x in np.unique(dock.current_mask())],
        'mask_counts': {str(int(x)): int(np.sum(dock.current_mask() == x))
                        for x in np.unique(dock.current_mask())},
        'mask_sha256': hashlib.sha256(dock.current_mask().tobytes()).hexdigest(),
        'slice': dock.current_slice(),
        'draft': dock.prompt.toPlainText(), 'x': point.x(), 'y': point.y(),
        'window_width': slicer.util.mainWindow().width,
        'window_height': slicer.util.mainWindow().height,
        'submit': center(dock.save_button), 'dialog': buttons,
        'interaction_visible': dock.hint_panel.visible,
        'interaction_target': dock.interaction_target,
        'interaction_mode': dock.interaction_mode,
        'hint_scope': dock.hint_scope_value,
        'model_selector': center(dock.models),
        'selected_model': dock.models.currentText,
        'target_selector': center(dock.hint_target.lineEdit()),
        'model_names': [dock.models.itemText(i) for i in range(dock.models.count)
                        if dock.models.itemData(i)],
        'model_groups': [dock.models.itemText(i) for i in range(dock.models.count)
                         if not dock.models.itemData(i) and dock.models.itemText(i)],
        'group_selectable': [bool(dock.models.model().item(i).isSelectable())
                             for i in range(dock.models.count) if not dock.models.itemData(i)],
        'slice_view_selector': center(dock.views),
        'slice_view': dock.slice_view,
        'slice_view_actions': {
            str(action.data()): [
                dock.view_menu.mapToGlobal(dock.view_menu.actionGeometry(action).center()).x(),
                dock.view_menu.mapToGlobal(dock.view_menu.actionGeometry(action).center()).y(),
            ] for action in dock.view_menu.actions()
        } if dock.view_menu.visible else {},
        'hint_separator_visible': dock.hint_separator.visible,
        'update_menu': [dock.update_hints.mapToGlobal(qt.QPoint(
            dock.update_hints.width - 7, dock.update_hints.height // 2)).x(),
            dock.update_hints.mapToGlobal(qt.QPoint(
            dock.update_hints.width - 7, dock.update_hints.height // 2)).y()],
        'visible_targets': [node.GetAttribute('MONAILabel.Target') for node in dock.region_nodes
                            if node.GetScene() and node.GetDisplayNode().GetVisibility()],
        'scope_actions': {
            str(action.data()): [
                dock.hint_scope_menu.mapToGlobal(dock.hint_scope_menu.actionGeometry(action).center()).x(),
                dock.hint_scope_menu.mapToGlobal(dock.hint_scope_menu.actionGeometry(action).center()).y(),
            ] for action in dock.hint_scope_menu.actions()
        } if dock.hint_scope_menu.visible else {},
        'placing': slicer.app.applicationLogic().GetInteractionNode().GetCurrentInteractionMode(),
        'pending_points': (dock.hint_placement[0].GetNumberOfDefinedControlPoints()
                           if getattr(dock, 'hint_placement', None) else -1),
        'spatial_objects': spatial_hints.inventory(dock)[0],
        'hint_buttons': {str(b.accessibleName): center(b) for b in
                         dock.hint_panel.findChildren(qt.QToolButton) if b.visible},
        'view_center': center(slicer.app.layoutManager().sliceWidget('Red').sliceView()),
        'history': dock.history.toPlainText(),
    }))
desktop_observer = qt.QTimer()
desktop_observer.setInterval(300)
desktop_observer.timeout.connect(observe_desktop)
desktop_observer.start()
"""

QUPATH_PROBE = """
        def observer = new javafx.animation.Timeline()
        observer.cycleCount = javafx.animation.Timeline.INDEFINITE
        observer.keyFrames.add(new javafx.animation.KeyFrame(
            javafx.util.Duration.millis(300), { event ->
            if (!panel?.asset || panel.busy || !panel.imageData) return
            def point = panel.prompt.localToScreen(panel.prompt.width / 2, panel.prompt.height / 2)
            if (point == null) return
            def view = panel.qupath.viewer.view
            def bounds = view.localToScreen(view.boundsInLocal)
            def center = { widget ->
                def p = widget.localToScreen(widget.width / 2, widget.height / 2)
                [p.x, p.y]
            }
            new File('/session/observed.json').text = Backend.json.toJson([
                asset_id: panel.asset.id, revision: panel.asset.revision,
                mode: panel.reviewMode ? 'review' : 'annotation',
                objects: panel.imageData.hierarchy.annotationObjects.size(),
                draft: panel.prompt.text, x: point.x, y: point.y,
                models: panel.modelChoice.items.findAll { it.id }.collect { it.name },
                selected_model: panel.modelChoice.value?.name,
                model_selector: center(panel.modelChoice),
                model_groups: panel.modelChoice.items.findAll { it.heading }.collect { it.name },
                window_width: panel.prompt.scene.window.width,
                window_height: panel.prompt.scene.window.height,
                image_center: [bounds.minX + bounds.width / 2, bounds.minY + bounds.height / 2],
                selected: panel.imageData.hierarchy.selectionModel.selectedObjects.size(),
                target: panel.targetChoice.editor.text,
                input_mode: panel.hints.mode,
                inputs: panel.hints.objects,
                input_buttons: panel.inputTools.children.collectEntries { b ->
                    [(b.userData): center(b)]
                },
                update: center(panel.updateButton),
                update_options: { def p = panel.updateButton.localToScreen(
                    panel.updateButton.width - 8, panel.updateButton.height / 2)
                    [p.x, p.y]
                }(),
                menu_actions: javafx.stage.Window.windows.findAll { it.showing }.collectMany { w ->
                    w.scene.root.lookupAll('.menu-item').toList()
                }.findAll { it.visible && it.localToScreen(0, 0) != null }.collectEntries { item ->
                    def text = item.lookup('.label')?.text
                    text ? [(text): center(item)] : [:]
                },
                target_selector: center(panel.targetChoice.editor),
                status: panel.status.text,
                history: panel.history.text,
                busy: panel.busy,
            ])
        } as javafx.event.EventHandler))
        observer.play()
"""


@pytest.fixture
def desktop_stack(tmp_path, monkeypatch, request):
    artifacts = ROOT / "test-results" / ("browser-desktop-" + secrets.token_hex(6))
    artifacts.mkdir(parents=True, mode=0o700)
    print(f"Browser desktop artifacts: {artifacts}", flush=True)
    bridges = tmp_path / "bridges"
    shutil.copytree(
        browser_desktop.RESOURCES, bridges, ignore=shutil.ignore_patterns("__pycache__")
    )
    with (bridges / "slicer_bridge.py").open("a") as stream:
        stream.write(SLICER_PROBE)
    extension = bridges / "qupath" / "MonaiLabelExtension.groovy"
    text = extension.read_text().replace(
        "void show(QuPathGUI qupath) {", "void show(QuPathGUI qupath) {" + QUPATH_PROBE
    )
    extension.write_text(text)
    monkeypatch.setattr(browser_desktop, "RESOURCES", bridges)
    monkeypatch.setenv("MONAILABEL_TOOLS_DIR", str(ROOT / "workspace/.cache/tools"))
    monkeypatch.setenv(
        "MONAILABEL_ALLOWED_HOSTS",
        "localhost,127.0.0.1,desktop.test," + os.environ.get("MONAILABEL_E2E_HOST", ""),
    )
    stack = VideoStack(tmp_path, artifacts)
    stack.https = getattr(request, "param", False)
    stack.start_workspace()
    if stack.https:
        monkeypatch.setenv("SSL_CERT_FILE", str(stack.root / "workspace/.tls/ca.crt"))
    try:
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            response = http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            )
            response.raise_for_status()
            if os.environ.get("MONAILABEL_E2E_LAN_ONLY") == "1":
                stack.stop_workspace()
                stack.start_workspace(lan_only=True)
                http.base_url = stack.url
                http.post(
                    "/api/auth/login",
                    json={"username": stack.username, "password": stack.password},
                ).raise_for_status()
            # A non-loopback workspace hostname exercises automatic browser launch.
            stack.url = stack.url.replace(
                "127.0.0.1", os.environ.get("MONAILABEL_E2E_HOST", "desktop.test")
            )
            yield stack, Client(http=http)
    finally:
        service = stack.app.state.services
        for session in service.store.list(DesktopSession):
            runtime = service.desktops.runtime
            try:
                completed = subprocess.run(
                    ["docker", "logs", runtime.name(session.id)], capture_output=True, text=True
                )
                log = completed.stdout + completed.stderr
                (artifacts / f"{session.viewer}.log").write_text(log)
            except Exception:
                pass
            # Serialize teardown with a close-tab timer already removing this container.
            service.desktops.end(session.id, service.store.get(User, session.user_id))
        stack.stop_workspace()


def observed(path, predicate, page, timeout=120):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text())
            if predicate(value):
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        page.wait_for_timeout(300)
    raise AssertionError(f"Native viewer did not reach the expected state: {path}\n{value}")


def fitted_desktop(page):
    from playwright.sync_api import expect

    canvas = page.frame_locator("#display").locator("#noVNC_container canvas")
    page.wait_for_function(
        """() => {
            const frame = document.querySelector('#display');
            const bounds = frame.getBoundingClientRect();
            const remote = frame.contentWindow;
            const canvas = frame.contentDocument.querySelector('#noVNC_container canvas');
            return Math.abs(bounds.width - innerWidth) < 1
                && Math.abs(bounds.height - innerHeight) < 1
                && canvas?.width === remote.innerWidth && canvas?.height === remote.innerHeight;
        }""",
        timeout=15000,
    )
    size = canvas.evaluate("({width: innerWidth, height: innerHeight})")
    # Canvas pixel dimensions come from the server's framebuffer, not CSS stretching.
    expect(canvas).to_have_attribute("width", str(size["width"]), timeout=15000)
    expect(canvas).to_have_attribute("height", str(size["height"]), timeout=15000)
    expect(canvas).to_be_visible()
    bounds = canvas.bounding_box()
    viewport = page.evaluate("({width: innerWidth, height: innerHeight})")
    assert bounds["x"] == 0 and bounds["y"] == 0
    assert abs(bounds["width"] - viewport["width"]) < 1
    assert abs(bounds["height"] - viewport["height"]) < 1
    return canvas


def settled_input(path, page):
    previous, since = None, time.monotonic()

    def settled(value):
        nonlocal previous, since
        position = (value["x"], value["y"], value["window_width"], value["window_height"])
        if position != previous:
            previous, since = position, time.monotonic()
        # JavaFX can lay out its split panes after the window resize notification.
        return time.monotonic() - since >= 1

    return observed(path, settled, page, timeout=15)


@pytest.mark.parametrize(
    "viewer,mode,mobile", [("slicer", "review", False), ("qupath", "annotation", True)]
)
def test_browser_native_viewer(desktop_stack, viewer, mode, mobile):
    from playwright.sync_api import expect, sync_playwright

    stack, client = desktop_stack
    setup = client.post("/api/demo")
    project_id = setup["project_id"]
    with stack.app.state.services.store.transaction() as session:
        session.insert(
            ModelRecord(
                project_id=project_id,
                name="VISTA3D",
                provider="vista3d",
                label_ids=[0],
                read_only=True,
            )
        )
    asset = client.get(f"/api/projects/{project_id}/assets")[0]
    if viewer == "qupath":
        source = io.BytesIO()
        Image.new("RGB", (256, 256), (220, 170, 190)).save(source, format="PNG")
        response = client.http.post(
            f"/api/projects/{project_id}/assets/upload",
            params={"group_id": "desktop-slide", "name": "Browser slide.png"},
            content=source.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        asset = response.json()
        mask = [
            [1 if (x - 128) ** 2 + (y - 128) ** 2 < 900 else 0 for x in range(256)]
            for y in range(256)
        ]
    else:
        mask = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask, "covered_labels": [0, 1, 2]},
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        options = (
            playwright.devices["iPad Pro 11"]
            if mobile
            else {"viewport": {"width": 1600, "height": 1100}}
        )
        context = browser.new_context(**options)
        page = context.new_page()
        errors = []
        context.on("page", lambda p: p.on("pageerror", lambda e: errors.append(str(e))))
        try:
            login_workspace(page, stack)
            page.goto(
                f"{stack.url}/{'reviews' if mode == 'review' else 'datasets'}?project={project_id}"
            )
            if viewer == "qupath":
                page.get_by_placeholder("Search samples, patients, slides or procedures").fill(
                    "Browser slide"
                )
            row = page.locator("#content tr").filter(has_text=asset["name"])
            with page.expect_popup(timeout=10000) as popup:
                row.get_by_role(
                    "button", name="Slicer" if viewer == "slicer" else "QuPath", exact=True
                ).click()
            desktop = popup.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            assert desktop.evaluate("isSecureContext") is False
            expect(desktop.locator("header")).not_to_be_visible()
            expect(desktop.locator("#display")).to_be_visible()
            bounds = desktop.locator("#display").bounding_box()
            assert bounds["y"] == 0
            assert abs(bounds["height"] - desktop.evaluate("innerHeight")) < 1
            identifier = desktop.url.rsplit("/", 1)[-1]
            runtime = stack.app.state.services.desktops.runtime
            report = runtime.root / identifier / "observed.json"
            frame = desktop.frame_locator("#display")
            expect(frame.locator("html")).to_have_class(
                re.compile("noVNC_connected"), timeout=30000
            )
            canvas = fitted_desktop(desktop)
            width, height = int(canvas.get_attribute("width")), int(canvas.get_attribute("height"))
            value = observed(
                report,
                lambda v: (
                    v["revision"] == 1
                    and (v.get("objects", 1) > 0)
                    and width - 10 <= v["window_width"] <= width
                    and height - 40 <= v["window_height"] <= height
                ),
                desktop,
            )
            assert value["asset_id"] == asset["id"]
            assert value["mode"] == mode
            if viewer == "slicer":
                assert value["shared_filesystem"] is False
                assert value["mask_labels"] == [0, 1, 2]
            else:
                assert "VISTA3D" not in value["models"]
                assert value["models"] and "Automatic" not in value["models"]
            value = settled_input(report, desktop)
            bounds = canvas.bounding_box()
            x = bounds["x"] + value["x"] * bounds["width"] / width
            y = bounds["y"] + value["y"] * bounds["height"] / height
            (stack.artifacts / "input-position.json").write_text(
                json.dumps({"native": value, "canvas": bounds, "tap": [x, y]})
            )
            if mobile:
                desktop.touchscreen.tap(x, y)
                frame.locator("#noVNC_control_bar_handle").click()
                expect(frame.locator("#noVNC_keyboard_button")).to_be_visible()
                frame.locator("#noVNC_keyboard_button").click()
                expect(frame.locator("#noVNC_keyboardinput")).to_be_focused()
            else:
                desktop.mouse.click(x, y)
            desktop.keyboard.type("My browser draft")
            observed(report, lambda v: v["draft"] == "My browser draft", desktop, timeout=15)
            if "noVNC_open" not in frame.locator("#noVNC_control_bar").get_attribute("class"):
                frame.locator("#noVNC_control_bar_handle").click()
            frame.locator("#noVNC_clipboard_button").click()
            clipboard = frame.locator("#noVNC_clipboard_text")
            clipboard.fill(" — clipboard α")
            frame.get_by_role("button", name="Paste into viewer", exact=True).click()
            draft = "My browser draft — clipboard α"
            observed(report, lambda v: v["draft"] == draft, desktop, timeout=15)
            desktop.keyboard.press("Control+a")
            desktop.keyboard.press("Control+c")
            expect(clipboard).to_have_value(draft)
            if "noVNC_open" not in frame.locator("#noVNC_control_bar").get_attribute("class"):
                frame.locator("#noVNC_control_bar_handle").click()
            frame.locator("#noVNC_clipboard_button").click()
            frame.get_by_role("button", name="Copy to computer", exact=True).click()
            expect(
                frame.get_by_role("status").filter(has_text="Copied to your computer.")
            ).to_be_visible()
            outside = context.new_page()
            outside.set_content('<textarea aria-label="Computer clipboard"></textarea>')
            outside_text = outside.get_by_role("textbox")
            outside_text.focus()
            outside.keyboard.press("Control+v")
            expect(outside_text).to_have_value(draft)
            outside_text.fill(" + keyboard paste")
            outside.keyboard.press("Control+a")
            outside.keyboard.press("Control+c")
            desktop.bring_to_front()
            frame.locator("#noVNC_clipboard_button").click()
            canvas.focus()
            desktop.keyboard.press("Control+End")
            desktop.keyboard.press("Control+v")
            draft += " + keyboard paste"
            observed(report, lambda v: v["draft"] == draft, desktop, timeout=15)
            outside.close()
            for size in (
                [{"width": 1194, "height": 834}, {"width": 834, "height": 1194}]
                if mobile
                else [{"width": 1920, "height": 1080}, {"width": 1280, "height": 900}]
            ):
                desktop.set_viewport_size(size)
                resized = fitted_desktop(desktop)
                remote = resized.evaluate("({width: innerWidth, height: innerHeight})")
                observed(
                    report,
                    lambda v, remote=remote: (
                        v["draft"] == draft
                        and remote["width"] - 10 <= v["window_width"] <= remote["width"]
                        and remote["height"] - 40 <= v["window_height"] <= remote["height"]
                    ),
                    desktop,
                    timeout=15,
                )
            desktop.screenshot(
                path=str(stack.artifacts / f"{viewer}-browser.png"), animations="disabled"
            )
            desktop.reload()
            expect(desktop.frame_locator("#display").locator("html")).to_have_class(
                re.compile("noVNC_connected"), timeout=30000
            )
            fitted_desktop(desktop)
            observed(report, lambda v: v["draft"] == draft, desktop, timeout=15)
            original_tab = desktop
            result = client.wait(
                client.post(
                    f"/api/assets/{asset['id']}/viewer?name={viewer}&mode={mode}&target=browser"
                )["id"]
            )
            assert result["url"] == f"/desktop/{identifier}"
            if viewer == "qupath":
                stack.app.state.services.assistants.provider.queue.append(
                    ChatMessage(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="open-desktop", name="open_viewer", arguments={"viewer": viewer}
                            )
                        ],
                    )
                )
                # Chat should still open automatically when the browser blocks popups.
                page.evaluate("window.open = () => null")
                if not page.locator("#chat-input").is_visible():
                    page.get_by_role("button", name="Assistant", exact=True).click()
                page.locator("#chat-input").fill("Open this sample in QuPath")
                page.locator('#chat-form [type="submit"]').click()
                page.wait_for_url(stack.url + result["url"], timeout=30000)
                desktop = page
            else:
                desktop = context.new_page()
                desktop.goto(stack.url + result["url"])
            expect(desktop.frame_locator("#display").locator("html")).to_have_class(
                re.compile("noVNC_connected"), timeout=30000
            )
            observed(report, lambda v: v["draft"] == draft, desktop, timeout=15)
            original_tab.close()
            expect(desktop.locator("header")).not_to_be_visible()
            desktop.close()
            # Allow the grace period and Docker's bounded removal under I/O load.
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                session = stack.app.state.services.store.get(DesktopSession, identifier)
                if session.ended:
                    break
                time.sleep(0.1)
            assert session.ended, "Closing the last viewer tab did not end its session"
            assert not runtime.running(identifier)
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 1
            assert not errors
        finally:
            for i, window in enumerate(context.pages):
                window.screenshot(path=str(stack.artifacts / f"page-{i}.png"))
                (stack.artifacts / f"page-{i}.html").write_text(window.content())
            (stack.artifacts / "browser-errors.json").write_text(json.dumps(errors))
            context.close()
            browser.close()


def test_slicer_radiology_quickstart_annotate_correct_and_submit(desktop_stack, monkeypatch):
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    service = stack.app.state.services
    project = client.post(
        "/api/projects",
        {
            "name": "Slicer radiology Quickstart",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
        },
    )
    values = np.zeros((24, 28, 12), dtype=np.int16)
    values[4:13, 7:21, 2:8] = 100
    response = client.http.post(
        f"/api/projects/{project['id']}/assets/upload",
        params={"name": "Slicer Quickstart.nii.gz", "group_id": "synthetic-slicer-ct"},
        content=gzip.compress(nib.Nifti1Image(values, np.diag([-1.5, 2, 2.5, 1])).to_bytes()),
    )
    response.raise_for_status()
    asset = response.json()
    calls = []

    class AnnotationFixture:
        def predict(self, image, labels, prompt, model):
            calls.append((model.name, image.shape))
            return Prediction((image[..., 0] > 0.5).astype(np.uint8))

    monkeypatch.setattr(service.models.recipes, "segmenter", lambda *_: AnnotationFixture())
    service.models.providers["openai-chat-polygons"] = AnnotationFixture()
    with service.store.transaction() as session:
        for name, provider in [("VISTA3D", "vista3d"), ("GPT Astra", "openai-chat-polygons")]:
            session.insert(
                ModelRecord(
                    project_id=project["id"],
                    name=name,
                    provider=provider,
                    config={"url": "http://unused.test", "model": "fixture"}
                    if provider == "openai-chat-polygons"
                    else {},
                    label_ids=[0, 1],
                    read_only=True,
                )
            )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="Slicer", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            canvas = fitted_desktop(desktop)
            identifier = desktop.url.rsplit("/", 1)[-1]
            report = service.desktops.runtime.root / identifier / "observed.json"
            observed(report, lambda value: value["asset_id"] == asset["id"], desktop)

            def prompt(message, tool=None, arguments=None):
                if tool:
                    service.assistants.provider.queue.append(
                        ChatMessage(
                            role="assistant",
                            tool_calls=[
                                ToolCall(id=secrets.token_hex(8), name=tool, arguments=arguments)
                            ],
                        )
                    )
                value = settled_input(report, desktop)
                bounds = canvas.bounding_box()
                desktop.mouse.click(
                    bounds["x"] + value["x"] * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"]
                    + value["y"] * bounds["height"] / int(canvas.get_attribute("height")),
                )
                desktop.keyboard.type(message)
                desktop.keyboard.press("Control+Enter")

            def mask_is(expected):
                digest = hashlib.sha256(expected.tobytes()).hexdigest()
                return observed(
                    report, lambda value: value["mask_sha256"] == digest, desktop, timeout=60
                )

            prompt(
                "Segment the spleen in the whole volume using VISTA3D.",
                "annotate",
                {"targets": ["spleen"], "scope": "full", "model_name": "VISTA3D"},
            )
            full = (values > 0).astype(np.uint8)
            current = mask_is(full)["slice"]
            assert current is not None
            cleared = full.copy()
            selected = [slice(None)] * 3
            selected[current["axis"]] = current["index"]
            cleared[tuple(selected)] = 0
            prompt(
                "Clear the spleen annotation on the current slice.",
                "clear_segments",
                {"targets": ["spleen"], "scope": "current_slice"},
            )
            mask_is(cleared)
            model_calls = len(service.assistants.provider.calls)
            prompt("Undo that.")
            mask_is(full)
            prompt("Redo that.")
            mask_is(cleared)
            assert len(service.assistants.provider.calls) == model_calls
            assert not service.assistants.provider.queue
            prompt(
                "Annotate the spleen on the current slice using GPT Astra.",
                "annotate",
                {"targets": ["spleen"], "scope": "current_slice", "model_name": "GPT Astra"},
            )
            mask_is(full)
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
            prompt("Submit this annotation for review.", "viewer_edit", {"operation": "submit"})
            observed(report, lambda value: value["revision"] == 1, desktop, timeout=30)
            saved = client.get(f"/api/assets/{asset['id']}")
            content = client.http.get(f"/api/annotations/{saved['annotation_id']}/mask.bin").content
            assert content == full.tobytes()
            assert calls[0] == ("VISTA3D", (*values.shape, 1))
            assert calls[1][0] == "GPT Astra" and len(calls[1][1]) == 3
            desktop.screenshot(path=str(stack.artifacts / "slicer-quickstart-submitted.png"))
        finally:
            for index, window in enumerate(context.pages):
                if not window.is_closed():
                    window.screenshot(path=str(stack.artifacts / f"quickstart-page-{index}.png"))
            context.close()
            browser.close()


def test_slicer_submit_exit_closes_tab_and_returns_blocked_tab(desktop_stack):
    from playwright.sync_api import expect, sync_playwright

    stack, client = desktop_stack
    setup = client.post("/api/demo")
    project_id = setup["project_id"]
    asset = client.get(f"/api/projects/{project_id}/assets")[0]
    mask = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
    client.post(
        f"/api/assets/{asset['id']}/review",
        {"base_revision": 0, "mask": mask, "covered_labels": [0, 1, 2]},
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project_id}")
            row = page.locator("#content tr").filter(has_text=asset["name"])
            with page.expect_popup() as opened:
                row.get_by_role("button", name="Slicer", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            canvas = fitted_desktop(desktop)
            identifier = desktop.url.rsplit("/", 1)[-1]
            service = stack.app.state.services
            report = service.desktops.runtime.root / identifier / "observed.json"
            observed(report, lambda value: value["revision"] == 1, desktop)
            value = settled_input(report, desktop)

            # A second, manually opened tab must return to the project if close is blocked.
            copied = context.new_page()
            copied.add_init_script("window.close = () => {}")
            copied.goto(desktop.url)
            fitted_desktop(copied)
            desktop.bring_to_front()

            def click_native(point):
                bounds = canvas.bounding_box()
                width = int(canvas.get_attribute("width"))
                height = int(canvas.get_attribute("height"))
                desktop.mouse.click(
                    bounds["x"] + point[0] * bounds["width"] / width,
                    bounds["y"] + point[1] * bounds["height"] / height,
                )

            click_native(value["submit"])
            dialog = observed(report, lambda value: "Exit Slicer" in value["dialog"], desktop)
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 2
            desktop.screenshot(path=str(stack.artifacts / "slicer-submitted.png"))
            with desktop.expect_event("close", timeout=30000):
                click_native(dialog["dialog"]["Exit Slicer"])
            copied.wait_for_url(f"{stack.url}/datasets?project={project_id}", timeout=30000)
            expect(copied.locator("#workspace")).to_be_visible()
            assert not page.is_closed()
            assert service.store.get(DesktopSession, identifier).ended
            assert not service.desktops.runtime.running(identifier)
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 2
        finally:
            for index, window in enumerate(context.pages):
                window.screenshot(path=str(stack.artifacts / f"exit-page-{index}.png"))
            context.close()
            browser.close()


def test_pathology_quickstart_region_annotation_and_submission(desktop_stack):
    from playwright.sync_api import sync_playwright

    stack, client = desktop_stack
    service = stack.app.state.services
    service.dataset_templates.downloads = Downloads(ROOT / "workspace/.cache/datasets")

    def chat(message, tool, arguments, project_id=None):
        service.assistants.provider.queue.append(
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id=secrets.token_hex(8), name=tool, arguments=arguments)],
            )
        )
        reply = client.post("/api/assistant", {"message": message, "project_id": project_id})
        return client.wait(reply["job_id"], timeout=120) if reply["job_id"] else reply["data"]

    project = chat('Create a project called "Pathology".', "create_project", {"name": "Pathology"})[
        "project"
    ]
    imported = chat(
        "Import the OpenSlide pathology sample.",
        "import_dataset_template",
        {"template_id": "openslide-cmu-small"},
        project["id"],
    )
    asset = client.get(f"/api/assets/{imported['asset_ids'][0]}")
    client.post(
        f"/api/projects/{project['id']}/models",
        {
            "name": "GPT Astra",
            "provider": "openai-chat-polygons",
            "config": {"url": "http://unused.test", "model": "fixture"},
        },
    )
    predictions = []

    class NucleiFixture:
        def predict(self, image, labels, prompt, model):
            predictions.append(image.shape)
            mask = np.zeros(image.shape[:-1], np.uint8)
            h, w = mask.shape
            mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = next(
                label.id for label in labels if label.id
            )
            return Prediction(mask)

    service.models.providers["openai-chat-polygons"] = NucleiFixture()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"]
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(f"{stack.url}/datasets?project={project['id']}")
            with page.expect_popup() as opened:
                page.get_by_role("button", name="QuPath", exact=True).click()
            desktop = opened.value
            desktop.wait_for_url("**/desktop/*", timeout=240000)
            canvas = fitted_desktop(desktop)
            identifier = desktop.url.rsplit("/", 1)[-1]
            report = service.desktops.runtime.root / identifier / "observed.json"
            observed(report, lambda value: value["asset_id"] == asset["id"], desktop)
            value = settled_input(report, desktop)

            def point(x, y):
                bounds = canvas.bounding_box()
                return (
                    bounds["x"] + x * bounds["width"] / int(canvas.get_attribute("width")),
                    bounds["y"] + y * bounds["height"] / int(canvas.get_attribute("height")),
                )

            x, y = value["image_center"]
            desktop.mouse.click(*point(x, y))
            desktop.keyboard.press("r")  # QuPath's native rectangle tool.
            desktop.mouse.move(*point(x - 50, y - 40))
            desktop.mouse.down()
            desktop.mouse.move(*point(x + 50, y + 40), steps=10)
            desktop.mouse.up()
            observed(report, lambda value: value["selected"] == 1, desktop, timeout=15)

            def prompt(message, tool, arguments):
                service.assistants.provider.queue.append(
                    ChatMessage(
                        role="assistant",
                        tool_calls=[
                            ToolCall(id=secrets.token_hex(8), name=tool, arguments=arguments)
                        ],
                    )
                )
                value = settled_input(report, desktop)
                desktop.mouse.click(*point(value["x"], value["y"]))
                desktop.keyboard.type(message)
                desktop.keyboard.press("Control+Enter")

            prompt(
                "Segment nuclei in the selected region using GPT Astra.",
                "annotate",
                {"targets": ["nuclei"], "scope": "selected_region", "model_name": "GPT Astra"},
            )
            observed(
                report,
                lambda value: value["status"].startswith("Applied nuclei/structure labels"),
                desktop,
                timeout=60,
            )
            assert len(predictions) == 1
            assert predictions[0][0] < asset["spatial_shape"][0]
            assert predictions[0][1] < asset["spatial_shape"][1]
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
            prompt("Submit this annotation for review.", "viewer_edit", {"operation": "submit"})
            observed(report, lambda value: value["revision"] == 1, desktop, timeout=30)
            saved = client.get(f"/api/assets/{asset['id']}")
            assert any(
                client.http.get(f"/api/annotations/{saved['annotation_id']}/mask.bin").content
            )
            prefix = f"/api/projects/{project['id']}"
            units = client.get(prefix + "/review-units")
            assert len(units) == 1 and units[0]["scope"]["kind"] == "region"
            region = units[0]["scope"]["region"]
            assert (region["height"], region["width"]) == predictions[0][:2]
            client.post(
                f"/api/review-units/{units[0]['id']}/decision",
                {
                    "base_revision": 1,
                    "verdict": "accepted",
                },
            )
            labels = client.get(prefix)["labels"]
            target = next(label["id"] for label in labels if label["name"].casefold() == "nuclei")
            learner = client.post(
                prefix + "/learners",
                {
                    "name": "Nuclei U-Net",
                    "recipe": "monai-unet",
                    "label_ids": [0, target],
                    "config": {
                        "epochs": 1,
                        "steps_per_epoch": 2,
                        "patch_size": 16,
                        "channels": [4, 8, 16, 32],
                        "device": "cpu",
                    },
                },
            )
            trained = client.wait(
                client.post(prefix + f"/learners/{learner['id']}/train", {})["id"]
            )
            prompt(
                "Segment nuclei in the selected region using Nuclei U-Net.",
                "annotate",
                {"targets": ["nuclei"], "scope": "selected_region", "model_name": "Nuclei U-Net"},
            )
            proposals = service.store.list(Proposal, project["id"])
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not any(
                p.model_ids == [trained["model_id"]] for p in proposals
            ):
                desktop.wait_for_timeout(100)
                proposals = service.store.list(Proposal, project["id"])
            assert any(p.model_ids == [trained["model_id"]] for p in proposals)
            observed(
                report,
                lambda value: (
                    not value["busy"]
                    and value["status"].startswith("Applied nuclei/structure labels")
                ),
                desktop,
                timeout=60,
            )
        finally:
            for index, window in enumerate(context.pages):
                window.screenshot(path=str(stack.artifacts / f"pathology-page-{index}.png"))
            context.close()
            browser.close()


@pytest.mark.parametrize("desktop_stack", [True], indirect=True)
@pytest.mark.parametrize("viewer,mobile", [("slicer", False), ("qupath", True)])
def test_browser_native_voice_preserves_draft(desktop_stack, viewer, mobile):
    from playwright.sync_api import expect, sync_playwright

    stack, client = desktop_stack
    project = client.post("/api/demo")["project_id"]
    asset = client.get(f"/api/projects/{project}/assets")[0]
    if viewer == "qupath":
        image = io.BytesIO()
        Image.new("RGB", (256, 256), (220, 170, 190)).save(image, format="PNG")
        reply = client.http.post(
            f"/api/projects/{project}/assets/upload",
            params={"name": "Voice slide.png", "group_id": "voice-slide"},
            content=image.getvalue(),
        )
        reply.raise_for_status()
        asset = reply.json()
    result = client.wait(
        client.post(f"/api/assets/{asset['id']}/viewer?name={viewer}&target=browser")["id"],
        timeout=240,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel=os.environ.get("MONAILABEL_E2E_BROWSER_CHANNEL"),
            args=["--host-resolver-rules=MAP desktop.test 127.0.0.1", "--no-proxy-server"],
        )
        options = (
            playwright.devices["iPad Pro 11"]
            if mobile
            else {"viewport": {"width": 1600, "height": 1000}}
        )
        context = browser.new_context(**options, ignore_https_errors=True)
        context.add_init_script(path=str(ROOT / "tests/e2e/speech_fixture.js"))
        errors = []
        context.on("page", lambda p: p.on("pageerror", lambda e: errors.append(str(e))))
        try:
            page = context.new_page()
            login_workspace(page, stack)
            page.goto(stack.url + result["url"])
            canvas = fitted_desktop(page)
            identifier = result["url"].rsplit("/", 1)[-1]
            report = stack.app.state.services.desktops.runtime.root / identifier / "observed.json"
            observed(report, lambda v: v["asset_id"] == asset["id"], page)
            value = settled_input(report, page)
            bounds = canvas.bounding_box()
            page.mouse.click(
                bounds["x"] + value["x"] * bounds["width"] / int(canvas.get_attribute("width")),
                bounds["y"] + value["y"] * bounds["height"] / int(canvas.get_attribute("height")),
            )
            page.keyboard.type("Existing draft. ")
            frame = page.frame_locator("#display")
            frame.locator("#noVNC_control_bar_handle").click()
            frame.get_by_role("button", name="Dictate a prompt", exact=True).click()
            panel = frame.get_by_role("region", name="Voice input")
            transcript = panel.get_by_role("textbox", name="Dictated prompt")
            record = panel.get_by_role("button", name="Use microphone", exact=True)
            expect(record).to_be_enabled()
            assert panel.evaluate("isSecureContext") is True
            record.click()
            panel.evaluate("speechFixture.result('Annotate the spleen')")
            expect(transcript).to_have_value("Annotate the spleen")
            observed(report, lambda v: v["draft"] == "Existing draft. ", page, timeout=10)
            expect(panel.get_by_role("button", name="Insert into viewer")).to_be_disabled()
            panel.evaluate("speechFixture.end()")
            transcript.fill("Annotate the spleen on this slice.")
            page.screenshot(path=str(stack.artifacts / f"{viewer}-voice-review.png"))
            panel.get_by_role("button", name="Insert into viewer").click()
            observed(
                report,
                lambda v: v["draft"] == "Existing draft. Annotate the spleen on this slice.",
                page,
                timeout=15,
            )
            # Inserting text never sends the prompt or saves annotations.
            assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
            assert not stack.app.state.services.assistants.provider.calls
            frame.get_by_role("button", name="Dictate a prompt", exact=True).click()
            expect(transcript).to_have_value("")
            record.click()
            panel.evaluate("speechFixture.error('not-allowed')")
            expect(panel.get_by_role("status")).to_contain_text("denied")
            record.click()
            panel.evaluate("speechFixture.result('Keep this draft')")
            frame.get_by_role("button", name="Dictate a prompt", exact=True).click()
            panel.evaluate("speechFixture.result('Discard this late result')")
            frame.get_by_role("button", name="Dictate a prompt", exact=True).click()
            expect(transcript).to_have_value("Keep this draft")
            page.screenshot(path=str(stack.artifacts / f"{viewer}-voice.png"))
            assert not errors
        finally:
            context.close()
            browser.close()
