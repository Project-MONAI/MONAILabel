"""Real native viewers, input streaming and reconnects from a browser-only client."""

import io
import json
import re
import secrets
import shutil
import subprocess
import time

import httpx
import numpy as np
import pytest
from conftest import ROOT, VideoStack
from PIL import Image
from test_video_cvat import login_workspace

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord
from monailabel.core.ports import Prediction
from monailabel.server.dataset_downloads import Downloads
from monailabel.server.desktops.models import DesktopSession
from monailabel.viewers import browser_desktop

pytestmark = pytest.mark.desktop_e2e

SLICER_PROBE = """
# Test observation only; the production bridge above handles all viewer actions.
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
        'draft': dock.prompt.toPlainText(), 'x': point.x(), 'y': point.y(),
        'window_width': slicer.util.mainWindow().width,
        'window_height': slicer.util.mainWindow().height,
        'submit': center(dock.save_button), 'dialog': buttons,
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
            new File('/session/observed.json').text = Backend.json.toJson([
                asset_id: panel.asset.id, revision: panel.asset.revision,
                mode: panel.reviewMode ? 'review' : 'annotation',
                objects: panel.imageData.hierarchy.annotationObjects.size(),
                draft: panel.prompt.text, x: point.x, y: point.y,
                models: panel.modelChoice.items as List,
                window_width: panel.prompt.scene.window.width,
                window_height: panel.prompt.scene.window.height,
                image_center: [bounds.minX + bounds.width / 2, bounds.minY + bounds.height / 2],
                selected: panel.imageData.hierarchy.selectionModel.selectedObjects.size(),
                status: panel.status.text,
            ])
        } as javafx.event.EventHandler))
        observer.play()
"""


@pytest.fixture
def desktop_stack(tmp_path, monkeypatch):
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
    monkeypatch.setenv("MONAILABEL_ALLOWED_HOSTS", "localhost,127.0.0.1,desktop.test")
    stack = VideoStack(tmp_path, artifacts)
    stack.start_workspace()
    try:
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            response = http.post(
                "/api/auth/setup", json={"username": stack.username, "password": stack.password}
            )
            response.raise_for_status()
            # A non-loopback workspace hostname exercises automatic browser launch.
            stack.url = stack.url.replace("127.0.0.1", "desktop.test")
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
            runtime.stop(session.id)
        stack.stop_workspace()


def observed(path, predicate, page, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text())
            if predicate(value):
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        page.wait_for_timeout(300)
    raise AssertionError(f"Native viewer did not reach the expected state: {path}")


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
                assert value["models"][0] == "Automatic"
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
            deadline = time.monotonic() + 25
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
            "name": "GPT Sol",
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
                "Segment nuclei in the selected region using GPT Sol.",
                "annotate",
                {"targets": ["nuclei"], "scope": "selected_region", "model_name": "GPT Sol"},
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
        finally:
            for index, window in enumerate(context.pages):
                window.screenshot(path=str(stack.artifacts / f"pathology-page-{index}.png"))
            context.close()
            browser.close()
