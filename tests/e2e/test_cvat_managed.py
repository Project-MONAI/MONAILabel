"""First-launch managed CVAT, native drafts and real local model assistance."""

import os
import secrets
import shutil
import subprocess
import threading
import time
import zipfile
from contextlib import contextmanager
from io import BytesIO
from unittest.mock import patch

import httpx
import pytest
from conftest import VideoStack
from PIL import Image, ImageDraw
from test_video_cvat import (
    create_project,
    draw_rectangle,
    expect,
    frame,
    import_video,
    log_step,
    login_workspace,
    open_editor,
)

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord, Project

pytestmark = pytest.mark.video_e2e


@pytest.fixture(scope="module")
def video_stack(tmp_path_factory):
    root = tmp_path_factory.mktemp("managed-cvat-e2e")
    from pathlib import Path

    artifacts = Path("test-results") / ("managed-cvat-" + secrets.token_hex(6))
    artifacts.mkdir(parents=True)
    stack = VideoStack(root, artifacts)
    print(f"\nManaged CVAT artifacts: {artifacts}", flush=True)
    environment = dict(os.environ)
    for key in (
        "MONAILABEL_CVAT_URL",
        "MONAILABEL_CVAT_PUBLIC_URL",
        "MONAILABEL_CVAT_TOKEN",
        "MONAILABEL_CACHE_DIR",
    ):
        environment.pop(key, None)
    environment["MONAILABEL_PRELOAD_MODELS"] = "0"
    environment["MONAILABEL_ALLOWED_HOSTS"] = "127.0.0.1,cvat.test"
    with patch.dict(os.environ, environment, clear=True):
        try:
            stack.start_workspace()
            with httpx.Client(base_url=stack.url) as http:
                http.post(
                    "/api/auth/setup", json={"username": stack.username, "password": stack.password}
                ).raise_for_status()
            yield stack
        finally:
            manager = stack.app.state.services.video_editors.manager
            stack.stop_workspace()
            if (manager.root / "runtime.json").exists():
                stack.compose_project = manager.project
                stack.compose("down", "--volumes", "--remove-orphans", timeout=120)
            shutil.rmtree(root)


@pytest.mark.parametrize("hostname", ["127.0.0.1", "cvat.test"])
def test_managed_install_launch_and_track(
    video_stack, browser_session, video_http, synthetic_clip, hostname, request
):
    stack = video_stack
    # Restart chooses a new port; restore only the hostname at teardown.
    request.addfinalizer(lambda: setattr(stack, "url", stack.url.replace("cvat.test", "127.0.0.1")))
    stack.url = stack.url.replace("127.0.0.1", hostname)
    context, errors = browser_session
    page = context.new_page()
    failures = []
    page.on("response", lambda r: failures.append((r.status, r.url)) if r.status >= 400 else None)
    login_workspace(page, stack)
    assert page.evaluate("isSecureContext") == (hostname == "127.0.0.1")
    project_id = create_project(page, "Managed CVAT tool tracking")
    video = import_video(page, stack, project_id, synthetic_clip)
    launched, page = open_editor(page, video_http, video["id"], keep_page=True)
    assert launched["url"].startswith("/cvat/editor/")
    page.wait_for_url(stack.url + launched["url"])
    native = page.frame_locator("#editor")
    try:
        expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    except Exception:
        print("Failed requests:", failures)
        print("Page errors:", errors)
        print(page.locator("#editor").content_frame.locator("body").inner_text()[:3000])
        raise
    cvat_frame = page.frames[1]
    expect(native.locator(".cvat-spinner")).to_have_count(0)
    draw_rectangle(cvat_frame, "Grasper", [30, 40, 120, 150])
    with delay_tracking() as ready:
        send_action(
            page,
            stack,
            "Track this tool for 3 frames",
            "track_selected_video_tool",
            {"frame_count": 3},
        )
        wait_for_result(page, ready)
        # An unrelated manual edit while inference is pending makes that result stale.
        draw_rectangle(cvat_frame, "Scissors", [180, 50, 250, 130])
    expect(page.locator("#status")).to_contain_text("draft changed")
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    # Route chat deterministically; the resulting provider call still runs the real SAM model.
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="track-in-cvat",
                    name="track_selected_video_tool",
                    arguments={"frame_count": 3},
                )
            ],
        )
    )
    native.locator(".cvat-objects-sidebar-state-item").first.hover()
    page.locator("#prompt").fill("Track this tool for 3 frames")
    with page.expect_response(
        lambda r: r.url.endswith("/assistant") and r.request.method == "POST"
    ) as response:
        page.locator("#send").click()
    assert response.value.status == 200, response.value.text()
    assert response.value.json()["tools"] == ["track_selected_video_tool"]
    job = wait_job(video_http, response.value.json()["job_id"])
    proposal = video_http.get(
        f"/api/videos/{video['id']}/tracking-proposals/{job['video_proposal_id']}"
    ).json()
    assert proposal["request"]["label_id"] == 1
    expect(page.locator("#status")).to_contain_text("Annotation added to draft", timeout=10000)
    expect(page.locator("#apply, #discard, #proposal")).to_have_count(0)
    frame(cvat_frame, 1)
    page.locator("#submit").click()
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    document = video_http.get(f"/api/videos/{video['id']}/tracks").json()
    assert document["base_revision"] == 1
    assert [k["frame"] for k in document["document"]["tracks"][0]["keyframes"]] == [0, 1, 2, 3]
    assert len(document["document"]["tracks"]) == 2
    assert document["document"]["tracks"][1]["keyframes"][0]["box"] == pytest.approx(
        [180, 50, 250, 130], abs=1
    )
    page.locator("#review summary").click()
    page.locator("#accept").click()
    expect(page.locator("#status")).to_contain_text("revision accepted")
    log_step(
        stack,
        "managed launch, real SAM tracking, chat, stale draft protection and review",
    )
    stack.stop_workspace()
    stack.start_workspace()
    stack.url = stack.url.replace("127.0.0.1", hostname)
    page.goto(stack.url + launched["url"])
    expect(page.frame_locator("#editor").locator("#cvat_canvas_background")).to_be_visible(
        timeout=30000
    )
    expect(page.frame_locator("#editor").locator(".cvat-objects-sidebar-state-item")).to_have_count(
        2
    )
    after = page.evaluate(
        "async (id) => (await fetch('/api/videos/' + id + '/tracks')).json()", video["id"]
    )
    assert after == document
    log_step(stack, "managed restart preserves credentials, CVAT task and submitted tracks")
    assert not errors


@contextmanager
def delay_tracking():
    from monailabel.sam.video import SamVideoTracker

    ready, release = threading.Event(), threading.Event()
    track = SamVideoTracker.track

    def delayed(*args, **kwargs):
        result = track(*args, **kwargs)
        ready.set()
        assert release.wait(60), "Test did not release the completed tracking result"
        return result

    with patch.object(SamVideoTracker, "track", delayed):
        try:
            yield ready
        finally:
            release.set()


def wait_for_result(page, ready):
    deadline = time.monotonic() + 180
    while not ready.is_set() and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert ready.is_set(), "The tracker did not finish"


def wait_job(http, identifier):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        job = http.get(f"/api/jobs/{identifier}").json()
        assert job["status"] not in {"failed", "cancelled", "interrupted"}, job.get("error")
        if job["status"] == "succeeded":
            return job["result"]
        time.sleep(0.2)
    pytest.fail("Video operation timed out")


def send_action(page, stack, prompt, tool, arguments):
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="video-action", name=tool, arguments=arguments)],
        )
    )
    page.locator("#prompt").fill(prompt)
    with page.expect_response(
        lambda r: r.url.endswith("/assistant") and r.request.method == "POST"
    ) as response:
        page.locator("#send").click()
    assert response.value.status == 200, response.value.text()
    return response.value.json()


def ask_to_track(page, stack, *, workspace=False):
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="track-sixteen",
                    name="track_selected_video_tool",
                    arguments={"frame_count": 16},
                )
            ],
        )
    )
    page.locator("#chat-input" if workspace else "#prompt").fill("Track tool for 16 frames")
    with page.expect_response(
        lambda r: r.url.endswith("/assistant") and r.request.method == "POST"
    ) as response:
        page.locator('#chat-form [type="submit"]' if workspace else "#send").click()
    assert response.value.status == 200, response.value.text()
    return response.value.json()


def test_managed_real_tool_tracking_sample(
    video_stack, browser_session, video_http, detection_endpoint
):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Snare tracking with SAM 2")
    imported = video_http.post(
        f"/api/projects/{project_id}/dataset-imports",
        json={"template_id": "hyperkvasir-tool-tracking"},
    )
    imported.raise_for_status()
    video_id = wait_job(video_http, imported.json()["id"])["video_ids"][0]
    page.goto(stack.url + f"/datasets?project={project_id}")
    reply = ask_to_track(page, stack, workspace=True)
    assert "Tracking has not started" in reply["message"]
    launched = wait_job(video_http, reply["job_id"])
    page.wait_for_url(stack.url + launched["url"], timeout=10000)
    native = page.frame_locator("#editor")
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    expect(page.locator("#options, #tracking-options, #track")).to_have_count(0)
    expect(page.locator("#selection")).to_contain_text("No tool selected")
    assert {job["kind"] for job in video_http.get(f"/api/projects/{project_id}/jobs").json()} == {
        "dataset_import",
        "video_editor",
    }
    cvat_frame = page.frames[1]
    model = video_http.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "GPT-6 Astra",
            "provider": "openai-chat-polygons",
            "config": {"url": detection_endpoint.url, "model": "astra-fixture"},
        },
    ).json()
    with stack.app.state.services.store.transaction() as session:
        astra = session.get(ModelRecord, model["id"])
        session.update(astra.model_copy(update={"preset": "nvidia-astra"}))
    other_model = video_http.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "GPT-5.6 Sol",
            "provider": "openai-chat-polygons",
            "config": {"url": detection_endpoint.url, "model": "sol-fixture"},
        },
    ).json()
    page.reload()
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    cvat_frame = page.frames[1]
    expect(page.locator("#detection-model")).to_have_value("")
    expect(page.locator("#options, #tracking-options, #track")).to_have_count(0)
    expect(page.locator("#send")).to_be_enabled()
    detection_endpoint.polygon = [(420, 390), (445, 390), (600, 555), (580, 570)]
    coordinator = stack.app.state.services.assistants.provider
    if endpoint := os.environ.get("MONAILABEL_E2E_COORDINATOR_URL"):
        from monailabel.providers.chat.config import CoordinatorConfig
        from monailabel.providers.chat.http import HttpChat

        coordinator = HttpChat(
            CoordinatorConfig(
                provider="compatible",
                base_url=endpoint,
                model=os.environ["MONAILABEL_E2E_COORDINATOR_MODEL"],
                key_env=os.environ.get("MONAILABEL_E2E_COORDINATOR_KEY_ENV"),
                thinking=True,
                temperature=0,
                max_tokens=8192,
            )
        )

    def annotate(prompt, arguments):
        if not endpoint:
            coordinator.queue.append(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="annotate-snare",
                            name="find_and_track_video_tool",
                            arguments=arguments,
                        )
                    ],
                )
            )
        with patch.object(stack.app.state.services.assistants, "provider", coordinator):
            page.locator("#prompt").fill(prompt)
            with page.expect_response(
                lambda r: r.url.endswith("/assistant") and r.request.method == "POST",
                timeout=120000,
            ) as response:
                page.locator("#send").click()
            assert response.value.status == 200, response.value.text()
            reply = response.value.json()
        assert reply["tools"][-1] == "find_and_track_video_tool", reply
        result = wait_job(video_http, reply["job_id"])
        proposal = video_http.get(
            f"/api/videos/{video_id}/tracking-proposals/{result['video_proposal_id']}"
        ).json()
        expect(page.locator("#status")).to_contain_text("Annotation added to draft", timeout=10000)
        expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(1)
        assert proposal["detection"]["model_id"] == model["id"]
        return proposal

    page.locator("#detection-model").select_option(other_model["id"])
    single = annotate(
        "segment tool in this frame using GPT astra",
        {
            "output": "polygon",
            "scope": "current_frame",
            "model_name": "GPT-6 Astra",
        },
    )
    assert len(single["keyframes"]) == 1 and "points" in single["keyframes"][0]
    assert single["provider"] == "openai-chat-polygons"
    expect(page.locator("#detection-model")).to_have_value(other_model["id"])
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(0)
    page.locator("#detection-model").select_option("")
    located = annotate(
        "Locate the snare and track for 16 frames",
        {
            "output": "box",
            "frame_count": 16,
            "label_name": "Snare",
        },
    )
    assert len(located["keyframes"]) == 16
    assert all("box" in key and "points" not in key for key in located["keyframes"])
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(0)
    proposal = annotate(
        "Segment tool and track for 16 frames",
        {
            "output": "polygon",
            "frame_count": 16,
            "label_name": "Snare",
        },
    )
    assert proposal["request"]["frame_count"] == 16
    assert proposal["request"]["client_id"] is None
    assert proposal["detection"]["model_id"] == model["id"]
    assert [call["model"] for call in detection_endpoint.calls] == ["astra-fixture"] * 3
    expect(page.locator("#status")).to_contain_text("Annotation added to draft", timeout=10000)
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(1)
    assert [k["frame"] for k in proposal["keyframes"]] == list(range(16))
    assert any(k["points"] != proposal["keyframes"][0]["points"] for k in proposal["keyframes"][1:])
    frame(cvat_frame, 4)
    expect(page.locator("#selection")).to_contain_text("Frame 4 of", timeout=10000)
    expect(native.locator(".cvat_canvas_shape")).to_have_count(1)
    page.screenshot(path=stack.artifacts / "snare-tracking-draft.png", full_page=True)
    page.locator("#submit").click()
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    saved = video_http.get(f"/api/videos/{video_id}/tracks").json()
    assert saved["base_revision"] == 1
    assert [k["frame"] for k in saved["document"]["tracks"][0]["keyframes"]] == list(range(17))
    assert saved["document"]["tracks"][0]["keyframes"][-1]["outside"] is True
    assert errors == []
    log_step(
        stack,
        "tracking chat opens CVAT, segments the snare through vision HTTP fixture, "
        "tracks 16 frames with real SAM and adds a new native track",
    )


@pytest.mark.parametrize("shape", ["rectangle", "polygon"])
def test_chat_clear_frames_preserves_tracks_and_supports_undo(
    video_stack, browser_session, video_http, synthetic_clip, shape
):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project = create_project(page, "Clear annotations through chat")
    video = import_video(page, stack, project, synthetic_clip)
    opened, page = open_editor(page, video_http, video["id"], keep_page=True)
    native = page.frame_locator("#editor")
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    cvat_frame = page.frames[1]
    editor = video_http.get(f"/api/cvat/editors/{opened['editor_id']}").json()["editor"]
    if shape == "rectangle":
        draw_rectangle(cvat_frame, "Grasper", [30, 40, 100, 150])
    else:
        # Seed native polygon interpolation with different vertex counts.
        # Clearing itself still travels through chat and the real native editor.
        cvat_frame.evaluate(
            """async label => {
          const bridge = window.monaiVideo;
          await bridge.apply({
            request: {draft_signature: await bridge.snapshot(), client_id: null,
                      seed: {frame: 0}, output: 'polygon'},
            keyframes: [
              {frame: 0, points: [30,40,100,40,70,150], outside: false, occluded: false},
              {frame: 5, points: [40,50,110,50,110,160,40,160], outside: false, occluded: false}
            ]
          }, label);
        }""",
            editor["label_map"]["1"],
        )
    draw_rectangle(cvat_frame, "Scissors", [180, 50, 250, 130])
    page.locator("#save").click()
    expect(page.locator("#status")).to_have_text("CVAT draft saved.")
    task_path = f"/cvat-api/jobs/{editor['job_id']}/annotations"
    original = video_http.get(task_path).json()
    assert len(original["tracks"]) == 2
    original_polygons = {}
    if shape == "polygon":
        for index in (0, 1, 4, 5):
            frame(cvat_frame, index)
            expect(page.locator("#selection")).to_contain_text(f"Frame {index} of")
            original_polygons[index] = native.locator(
                "polygon.cvat_canvas_shape:visible"
            ).get_attribute("points")
    frame(cvat_frame, 2)
    expect(page.locator("#selection")).to_contain_text("Frame 2 of")
    send_action(
        page,
        stack,
        "Clear the grasper annotations for 2 frames.",
        "clear_video_annotations",
        {"targets": ["Grasper"], "scope": "frames", "frame_count": 2},
    )
    expect(page.locator("#status")).to_have_text("Draft updated. Submit when ready.")
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(1)
    for value, count in [(1, 2), (2, 1), (3, 1), (4, 2), (5, 2)]:
        frame(cvat_frame, value)
        expect(page.locator("#selection")).to_contain_text(f"Frame {value} of")
        expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(count)
        if shape == "polygon" and value in original_polygons:
            expect(native.locator("polygon.cvat_canvas_shape:visible")).to_have_attribute(
                "points", original_polygons[value]
            )
    page.locator("#save").click()
    expect(page.locator("#status")).to_have_text("CVAT draft saved.")
    cleared = video_http.get(task_path).json()
    assert {t["id"] for t in cleared["tracks"]} == {t["id"] for t in original["tracks"]}
    send_action(page, stack, "Undo that.", "viewer_edit", {"operation": "undo"})
    expect(page.locator("#status")).to_have_text("Undid the last edit.")
    frame(cvat_frame, 2)
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(2)
    send_action(page, stack, "Redo that.", "viewer_edit", {"operation": "redo"})
    expect(page.locator("#status")).to_have_text("Redid the last edit.")
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(1)
    send_action(page, stack, "Save my draft.", "viewer_edit", {"operation": "save_draft"})
    expect(page.locator("#status")).to_have_text("CVAT draft saved.")
    saved = video_http.get(task_path).json()
    assert len(saved["tracks"]) == 2
    scissors = original["tracks"][1]
    assert next(t for t in saved["tracks"] if t["id"] == scissors["id"]) == scissors
    page.reload()
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    cvat_frame = page.frames[1]
    frame(cvat_frame, 2)
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(1)
    frame(cvat_frame, 4)
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(2)
    send_action(
        page,
        stack,
        "Clear all annotations in the whole video.",
        "clear_video_annotations",
        {"all_targets": True, "scope": "whole_video"},
    )
    expect(page.locator("#status")).to_have_text("Draft updated. Submit when ready.")
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(0)
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat_canvas_shape:visible")).to_have_count(2)
    send_action(
        page, stack, "Submit this annotation for review.", "viewer_edit", {"operation": "submit"}
    )
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 1
    assert not errors
    log_step(
        stack,
        "CVAT chat scoped clear, preserved saved track IDs, native and chat undo, "
        "saved reload, submission",
    )
    if shape == "polygon":
        prefix = f"/api/projects/{project}"
        for unit in video_http.get(prefix + "/review-units").json():
            response = video_http.post(
                f"/api/review-units/{unit['id']}/decision",
                json={
                    "base_revision": unit["revision"],
                    "verdict": "accepted",
                },
            )
            assert response.status_code == 201, response.text
        response = video_http.post(
            prefix + "/learners",
            json={
                "name": "Grasper U-Net",
                "recipe": "monai-unet",
                "label_ids": [0, 1],
                "config": {
                    "epochs": 1,
                    "steps_per_epoch": 16,
                    "patch_size": 16,
                    "channels": [4, 8, 16, 32],
                    "device": "cpu",
                },
            },
        )
        assert response.status_code == 201, response.text
        learner = response.json()
        response = video_http.post(prefix + f"/learners/{learner['id']}/train", json={})
        assert response.status_code == 202, response.text
        trained = wait_job(video_http, response.json()["id"])
        # Reopen the submitted source and use the actual trained checkpoint in CVAT.
        page.goto(stack.url + f"/datasets?project={project}")
        _, page = open_editor(page, video_http, video["id"], keep_page=True)
        expect(page.frame_locator("#editor").locator("#cvat_canvas_background")).to_be_visible(
            timeout=30000
        )
        reply = send_action(
            page,
            stack,
            "Segment the grasper on this frame using Grasper U-Net.",
            "find_and_track_video_tool",
            {
                "output": "polygon",
                "scope": "current_frame",
                "label_name": "Grasper",
                "model_name": "Grasper U-Net",
            },
        )
        result = wait_job(video_http, reply["job_id"])
        assert result.get("video_proposal_id"), result
        proposal = video_http.get(
            f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
        ).json()
        assert proposal["detection"]["model_id"] == trained["model_id"]
        expect(page.locator("#status")).to_have_text(
            "Annotation added to draft. Use CVAT Undo to revert.", timeout=30000
        )
        assert not errors
        log_step(
            stack, "Accepted video polygons → actual U-Net training → native CVAT segmentation"
        )


def test_mixed_dataset_list(video_stack, browser_session, video_http, synthetic_clip):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Mixed image and video samples")

    def upload_image(index):
        content = BytesIO()
        Image.new("RGB", (32, 24), color=(index * 20, 90, 120)).save(content, format="PNG")
        response = video_http.post(
            f"/api/projects/{project_id}/assets/upload",
            params={"name": f"sample-{index}.png", "group_id": f"image-{index}"},
            content=content.getvalue(),
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 201, response.text
        return response.json()

    images = [upload_image(index) for index in range(5)]
    video = import_video(page, stack, project_id, synthetic_clip, group="Mixed Procedure")
    images.extend(upload_image(index) for index in range(5, 11))
    page.locator("#refresh").click()
    rows = page.locator("#content table tbody tr")
    pagination = page.locator("#content .pagination")
    selection = page.locator("#content .selection-bar")
    video_checkbox = page.locator(f'[data-file-selection="{video["id"]}"]')
    expect(page.locator("#content table")).to_have_count(1)
    expect(page.locator("#content .video-list")).to_have_count(0)
    expect(pagination).to_contain_text("1–10 of 12")
    expect(rows).to_have_count(10)
    expect(
        rows.filter(has_text=synthetic_clip.name).get_by_role("button", name="CVAT", exact=True)
    ).to_be_visible()

    page.locator("#select-all-files").check()
    expect(selection).to_contain_text("10 selected")
    expect(video_checkbox).to_be_checked()
    page.get_by_role("button", name="Next page", exact=True).click()
    expect(rows).to_have_count(2)
    expect(pagination).to_contain_text("11–12 of 12")
    expect(page.locator("[data-file-selection]:checked")).to_have_count(0)
    page.get_by_role("button", name="Previous page", exact=True).click()
    page.locator("#refresh").click()
    expect(selection).to_contain_text("10 selected")
    expect(video_checkbox).to_be_checked()
    page.get_by_role("button", name="Clear selection", exact=True).click()
    page.get_by_role("button", name="Select all 12 matching", exact=True).click()
    expect(selection).to_contain_text("12 selected")
    page.get_by_role("button", name="Clear selection", exact=True).click()

    search = page.get_by_role("searchbox", name="Search datasets")
    search.fill("  MIXED procedure  ")
    expect(rows).to_have_count(1)
    expect(rows).to_contain_text(synthetic_clip.name)
    expect(pagination).to_contain_text("1–1 of 1")
    search.fill("")
    page.locator("#dataset-filter").select_option("kind:video")
    expect(rows).to_have_count(1)
    expect(rows).to_contain_text(synthetic_clip.name)
    search.fill("no-such-sample")
    expect(page.locator("#content")).to_contain_text("No samples match these filters.")
    page.get_by_role("button", name="Reset filters", exact=True).click()
    expect(rows).to_have_count(10)
    page.get_by_role("button", name=f"Actions for {synthetic_clip.name}", exact=True).click()
    expect(page.locator("#dialog")).to_contain_text("Mixed Procedure")
    expect(page.locator("#dialog")).to_contain_text("6 frames")
    expect(page.get_by_role("button", name="Track data", exact=True)).to_be_visible()
    page.locator("#close-dialog").click()
    page.screenshot(path=stack.artifacts / "mixed-dataset-list.png", full_page=True)

    video_checkbox.check()
    page.locator(f'[data-file-selection="{images[0]["id"]}"]').check()
    page.get_by_role("button", name="Delete selected files", exact=True).click()
    expect(page.locator("#dialog .deletion-files li")).to_have_count(2)
    expect(page.locator("#dialog")).to_contain_text(synthetic_clip.name)
    expect(page.locator("#dialog")).to_contain_text(images[0]["name"])
    expect(page.locator("#dialog")).to_contain_text("CVAT tasks and saved drafts")
    with page.expect_response(
        lambda r: r.url.endswith(f"/projects/{project_id}/assets") and r.request.method == "DELETE"
    ) as deleted:
        page.locator("#action-form button[type=submit]").click()
    assert deleted.value.ok, deleted.value.text()
    expect(page.locator("#dialog")).not_to_be_visible()
    expect(rows).to_have_count(10)
    expect(pagination).to_contain_text("1–10 of 10")
    expect(page.locator("#content")).not_to_contain_text(synthetic_clip.name)
    remaining = video_http.get(f"/api/projects/{project_id}/assets").json()
    assert {asset["id"] for asset in remaining} == {asset["id"] for asset in images[1:]}
    assert video_http.get(f"/api/projects/{project_id}/videos").json() == []
    assert not errors
    log_step(stack, "mixed dataset search, filters, pagination, selection and bulk deletion")


def test_cvat_launch_failure_closed_tab_and_popup_fallback(
    video_stack, browser_session, synthetic_clip
):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "CVAT automatic launch controls")
    video = import_video(page, stack, project_id, synthetic_clip)
    endpoint = f"**/api/videos/{video['id']}/editor"
    page.route(
        endpoint,
        lambda route: route.fulfill(status=503, json={"detail": "CVAT launch test unavailable"}),
    )
    with page.expect_popup() as popup:
        page.get_by_role("button", name="CVAT", exact=True).click()
    failed = popup.value
    expect(page.locator("#messages")).to_contain_text("CVAT launch test unavailable")
    deadline = time.monotonic() + 5
    while not failed.is_closed() and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert failed.is_closed(), "A failed launch must close its unused loading tab"
    page.unroute(endpoint)

    pending = []
    page.route(endpoint, lambda route: pending.append(route))
    with page.expect_popup() as popup:
        page.get_by_role("button", name="CVAT", exact=True).click()
    closed = popup.value
    closed.close()
    deadline = time.monotonic() + 5
    while not pending and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert len(pending) == 1
    pending[0].continue_()
    page.unroute(endpoint)
    expect(page.get_by_role("link", name="Open CVAT annotation viewer")).to_be_visible(
        timeout=120000
    )
    assert page.url == f"{stack.url}/datasets?project={project_id}"

    # Simulate a browser blocking new windows; the existing tab still opens the editor.
    page.evaluate("window.open = () => null")
    page.get_by_role("button", name="CVAT", exact=True).click()
    page.wait_for_url("**/cvat/editor/*", timeout=30000)
    expect(page.frame_locator("#editor").locator("#cvat_canvas_background")).to_be_visible(
        timeout=30000
    )
    assert not errors
    log_step(
        stack,
        "failed launch closes loading tab, closed tab stays closed, blocked popup opens in place",
    )


def test_find_track_preserves_drafts_and_honors_named_model(
    video_stack, browser_session, video_http, synthetic_clip, detection_endpoint
):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Vision seeded native tracks")
    video = import_video(page, stack, project_id, synthetic_clip)
    models = []
    for name, remote in [("GPT-5.6 Sol", "sol-fixture"), ("GPT-6 Astra", "astra-fixture")]:
        response = video_http.post(
            f"/api/projects/{project_id}/models",
            json={
                "name": name,
                "provider": "openai-chat-polygons",
                "config": {"url": detection_endpoint.url, "model": remote},
            },
        )
        response.raise_for_status()
        models.append(response.json())
    _, page = open_editor(page, video_http, video["id"], keep_page=True)
    native = page.frame_locator("#editor")
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    cvat_frame = page.frames[1]
    draw_rectangle(cvat_frame, "Scissors", [180, 50, 250, 130])
    frame(cvat_frame, 2)
    page.locator("#detection-model").select_option(models[0]["id"])
    detection_endpoint.status, detection_endpoint.box = "not_found", None
    send_action(
        page,
        stack,
        "Locate the grasper and track for 3 frames",
        "find_and_track_video_tool",
        {"frame_count": 3, "label_name": "Grasper"},
    )
    expect(page.locator("#status")).to_contain_text("could not locate", timeout=10000)
    expect(page.locator("#proposal")).to_have_count(0)
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(1)
    detection_endpoint.status, detection_endpoint.box = "found", [30, 40, 120, 150]
    with delay_tracking() as ready:
        send_action(
            page,
            stack,
            "Locate the grasper and track for 3 frames",
            "find_and_track_video_tool",
            {"frame_count": 3, "label_name": "Grasper"},
        )
        wait_for_result(page, ready)
        draw_rectangle(cvat_frame, "Scissors", [150, 160, 200, 210])
    expect(page.locator("#status")).to_contain_text("draft changed")
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    stack.app.state.services.assistants.provider.queue.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="find-astra",
                    name="find_and_track_video_tool",
                    arguments={
                        "model_name": "GPT-6 Astra",
                        "frame_count": 3,
                        "label_name": "Grasper",
                    },
                ),
            ],
        )
    )
    page.locator("#prompt").fill("Use GPT-6 Astra to find the grasper and track it for 3 frames")
    page.locator("#send").click()
    expect(page.locator("#status")).to_contain_text("Annotation added to draft", timeout=180000)
    expect(page.locator("#messages")).to_contain_text("using GPT-6 Astra")
    assert [call["model"] for call in detection_endpoint.calls] == [
        "sol-fixture",
        "sol-fixture",
        "astra-fixture",
    ]
    expect(page.locator("#detection-model")).to_have_value(models[0]["id"])

    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(3)
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    native.locator(".cvat-annotation-header-redo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(3)
    page.locator("#submit").click()
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    tracks = video_http.get(f"/api/videos/{video['id']}/tracks").json()["document"]["tracks"]
    assert len(tracks) == 3
    assert tracks[0]["keyframes"][0]["box"] == pytest.approx([180, 50, 250, 130], abs=1)
    assert tracks[1]["keyframes"][0]["box"] == pytest.approx([150, 160, 200, 210], abs=1)
    assert [k["frame"] for k in tracks[2]["keyframes"]] == [2, 3, 4, 5]
    assert tracks[2]["keyframes"][-1]["outside"] is True
    assert errors == []
    log_step(
        stack,
        "find-and-track abstention, stale drafts, explicit model override, "
        "native undo/redo and unrelated tracks",
    )


def test_polygon_annotation_and_whole_video_tracking(
    video_stack, browser_session, video_http, detection_endpoint
):
    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    project = create_project(page, "Polygon tracking across chunks")
    # Clearly synthetic moving L-shaped object; exercise geometry, not model accuracy.
    directory = stack.root / "polygon-frames"
    directory.mkdir()
    polygon = [(40, 70), (120, 70), (120, 90), (65, 90), (65, 160), (40, 160)]
    for index in range(70):
        image = Image.new("RGB", (320, 240), "#c5c5c5")
        ImageDraw.Draw(image).polygon([(x + index // 2, y) for x, y in polygon], fill="#254d78")
        image.save(directory / f"{index:03d}.png")
    clip = stack.root / "synthetic-polygon.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            "10",
            "-i",
            str(directory / "%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    video = import_video(page, stack, project, clip, frames=70)
    model = video_http.post(
        f"/api/projects/{project}/models",
        json={
            "name": "GPT-6 Astra",
            "provider": "openai-chat-polygons",
            "config": {"url": detection_endpoint.url, "model": "polygon-fixture"},
        },
    ).json()
    service = stack.app.state.services
    with service.store.transaction() as session:
        astra = session.get(ModelRecord, model["id"])
        session.update(astra.model_copy(update={"preset": "nvidia-astra"}))
        vista = ModelRecord(
            project_id=project, name="VISTA3D", provider="vista3d", label_ids=[0], read_only=True
        )
        session.insert(vista)
        session.insert(
            astra.model_copy(
                update={"id": "other-vision", "name": "Other vision model", "preset": None}
            )
        )
        current = session.get(Project, project)
        session.update(current.model_copy(update={"annotation_model_id": vista.id}))
    _, page = open_editor(page, video_http, video["id"], keep_page=True)
    native = page.frame_locator("#editor")
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    expect(native.locator(".cvat-spinner")).to_have_count(0)
    cvat_frame = page.frames[1]
    expect(page.locator("#options, #tracking-options")).to_have_count(0)
    expect(page.locator("#review")).to_be_hidden()
    expect(page.locator("#messages")).to_be_visible()
    for width, height in [(900, 720), (1600, 1100)]:
        page.set_viewport_size({"width": width, "height": height})
        composer = page.locator("#chat").bounding_box()
        assert composer and composer["y"] + composer["height"] <= height
    expect(page.locator("#detection-model")).to_have_value("")
    frame(cvat_frame, 3)
    expect(page.locator("#selection")).to_contain_text("Frame 3 of")

    def ask(prompt, tool, arguments):
        stack.app.state.services.assistants.provider.queue.append(
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="polygon-action", name=tool, arguments=arguments)],
            )
        )
        page.locator("#prompt").fill(prompt)
        with page.expect_response(
            lambda r: r.url.endswith("/assistant") and r.request.method == "POST"
        ) as response:
            page.locator("#prompt").press("Control+Enter")
        assert response.value.status == 200, response.value.text()
        reply = response.value.json()
        result = wait_job(video_http, reply["job_id"])
        proposal = video_http.get(
            f"/api/videos/{video['id']}/tracking-proposals/{result['video_proposal_id']}"
        ).json()
        expect(page.locator("#status")).to_contain_text("Annotation added to draft", timeout=10000)
        expect(page.locator("#apply, #discard, #proposal")).to_have_count(0)
        return proposal

    detection_endpoint.polygon = [(x + 1, y) for x, y in polygon]
    automatic = ask(
        "Segment tool and track for 16 frames",
        "find_and_track_video_tool",
        {"output": "polygon", "frame_count": 16, "label_name": "Grasper"},
    )
    assert automatic["provider"] == "sam2"
    assert automatic["detection"]["model_id"] == model["id"]
    assert [key["frame"] for key in automatic["keyframes"]] == list(range(3, 19))
    assert all(len(key["points"]) >= 6 for key in automatic["keyframes"])
    expect(page.locator("#messages")).to_contain_text("using GPT-6 Astra")
    expect(page.locator("#detection-model")).to_have_value("")
    expect(native.locator("polygon.cvat_canvas_shape")).to_have_count(1)
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(0)
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    frame(cvat_frame, 0)
    draw_rectangle(cvat_frame, "Scissors", [180, 50, 250, 130])
    frame(cvat_frame, 3)
    detection_endpoint.box = [41, 70, 121, 160]
    single_box = ask(
        "Put a bounding box around the grasper on this frame",
        "find_and_track_video_tool",
        {"output": "box", "scope": "current_frame", "label_name": "Grasper"},
    )
    assert single_box["provider"] == "openai-chat-polygons"
    assert single_box["model_checksum"] is None
    assert single_box["keyframes"] == [
        {"frame": 3, "box": [41, 70, 121, 160], "outside": False, "occluded": False}
    ]
    expect(page.locator("#messages .assistant").last).to_contain_text("frame 3")
    expect(page.locator("#messages .assistant").last).not_to_contain_text("frames 3–3")
    native.locator(".cvat-annotation-header-undo-button").click()
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(1)
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    page.on("dialog", lambda dialog: dialog.accept())
    page.reload()
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(0)
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    cvat_frame = page.frames[1]
    draw_rectangle(cvat_frame, "Scissors", [180, 50, 250, 130])
    frame(cvat_frame, 3)
    expect(page.locator("#detection-model")).to_have_value("")
    detection_endpoint.polygon = [(x + 1, y) for x, y in polygon]
    single = ask(
        "Segment the grasper on this frame",
        "find_and_track_video_tool",
        {"output": "polygon", "scope": "current_frame", "label_name": "Grasper"},
    )
    assert single["provider"] == "openai-chat-polygons"
    assert single["request"]["seed"]["frame"] == 3
    assert len(single["keyframes"]) == 1
    assert len(single["keyframes"][0]["points"]) > 8
    expect(page.get_by_role("link", name="Download original masks")).to_be_hidden()
    page.get_by_text("Segmentation details", exact=True).click()
    expect(page.get_by_role("link", name="Download original masks")).to_be_visible()
    expect(native.locator("polygon.cvat_canvas_shape")).to_have_count(1)

    expect(native.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    native.locator(".cvat-objects-sidebar-state-item").last.hover()
    ranged = ask(
        "Track this polygon for 3 frames",
        "track_selected_video_tool",
        {"frame_count": 3},
    )
    assert ranged["request"]["output"] == "polygon"
    assert ranged["request"]["client_id"] is not None
    assert [key["frame"] for key in ranged["keyframes"]] == [3, 4, 5]
    assert ranged["keyframes"][0]["points"] == single["keyframes"][0]["points"]

    detection_endpoint.polygon = polygon
    whole = ask(
        "Segment the grasper and track it for the whole video",
        "find_and_track_video_tool",
        {"output": "polygon", "scope": "whole_video", "label_name": "Grasper"},
    )
    assert whole["provider"] == "sam2"
    assert whole["request"]["seed"]["frame"] == 0
    assert whole["request"]["frame_count"] == 70
    assert [key["frame"] for key in whole["keyframes"]] == list(range(70))
    assert not any(key["outside"] for key in whole["keyframes"][63:65])
    downloaded = video_http.get(f"/api/videos/{video['id']}/tracking-proposals/{whole['id']}/masks")
    downloaded.raise_for_status()
    with zipfile.ZipFile(BytesIO(downloaded.content)) as archive:
        assert len(archive.namelist()) == 71
        for index in (0, 63, 64, 69):
            mask = Image.open(BytesIO(archive.read(f"{index:06d}.png")))
            assert mask.size == (320, 240)
            assert mask.getbbox() is not None
    for index in (0, 64, 69):
        frame(cvat_frame, index)
        expect(page.locator("#selection")).to_contain_text(f"Frame {index} of")
        points = cvat_frame.evaluate("""async () => {
            const state = window.monaiVideo;
            return (await state.context()).frame;
        }""")
        assert points == index
        expect(native.locator("polygon.cvat_canvas_shape")).to_have_count(1)
    page.screenshot(path=stack.artifacts / "polygon-whole-video-draft.png", full_page=True)

    page.locator("#save").click()
    expect(page.locator("#status")).to_contain_text("CVAT draft saved")
    page.reload()
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    page.locator("#submit").click()
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    saved = video_http.get(f"/api/videos/{video['id']}/tracks").json()
    tracks = saved["document"]["tracks"]
    assert len(tracks) == 3
    assert tracks[0]["keyframes"][0]["box"] == pytest.approx([180, 50, 250, 130], abs=1)
    assert tracks[1]["keyframes"][:3] == ranged["keyframes"]
    assert tracks[2]["keyframes"] == whole["keyframes"]
    review = video_http.post(
        f"/api/videos/{video['id']}/editor", json={"base_revision": 1, "mode": "review"}
    )
    review.raise_for_status()
    launched = wait_job(video_http, review.json()["id"])
    page.goto(stack.url + launched["url"])
    expect(native.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    expect(page.locator("#accept")).to_be_visible()
    page.locator("#submit").click()
    expect(page.locator("#status")).to_contain_text("Tracks submitted", timeout=20000)
    reviewed = video_http.get(f"/api/videos/{video['id']}/tracks").json()
    assert reviewed["base_revision"] == 2
    assert reviewed["document"] == saved["document"]
    page.locator("#accept").click()
    expect(page.locator("#status")).to_contain_text("revision accepted")
    assert len(detection_endpoint.calls) == 4
    assert errors == []
    log_step(
        stack,
        "chat-focused panel, boxes and polygons on one frame, selected polygon "
        "tracking, whole-video chunk boundaries, source masks and mixed-shape review",
    )
