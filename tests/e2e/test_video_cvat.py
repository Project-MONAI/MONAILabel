"""Real browser-to-browser video workflow. Annotation writes go through CVAT's UI."""

import hashlib
import json
import re
import time
from urllib.parse import urljoin

import pytest

pytestmark = pytest.mark.video_e2e


def expect(locator):
    from playwright.sync_api import expect as assertion

    return assertion(locator)


def log_step(stack, name, **details):
    path = stack.artifacts / "checks.json"
    checks = json.loads(path.read_text()) if path.exists() else []
    checks.append({"check": name, "status": "passed", **details})
    path.write_text(json.dumps(checks, indent=2))
    print(f"PASS: {name}", flush=True)


def login_workspace(page, stack):
    page.goto(stack.url)
    page.locator("#login-form input[name=username]").fill(stack.username)
    page.locator("#login-form input[name=password]").fill(stack.password)
    page.locator("#login-button").click()
    expect(page.locator("#workspace")).to_be_visible()


def create_project(page, name):
    page.locator("#new-project").click()
    page.locator("#action-form input[name=name]").fill(name)
    with page.expect_response(
        lambda r: r.url.endswith("/api/projects") and r.request.method == "POST"
    ) as response:
        page.locator("#action-form button[type=submit]").click()
    assert response.value.status == 201
    expect(page.locator("#dialog")).not_to_be_visible()
    return response.value.json()["id"]


def import_video(
    page, stack, project_id, clip, group="synthetic-procedure", split="pool", frames=6
):
    page.goto(f"{stack.url}/datasets?project={project_id}")
    page.get_by_role("button", name="Import video", exact=True).click()
    page.locator("input[name=clip]").set_input_files(clip)
    page.locator("input[name=group_id]").fill(group)
    page.locator("input[name=labels]").fill("Grasper, Scissors")
    page.locator("#action-form select[name=split]").select_option(split)
    with page.expect_response(
        lambda r: "/videos/upload?" in r.url and r.request.method == "POST"
    ) as response:
        page.locator("#action-form button[type=submit]").click()
    assert response.value.status == 201, response.value.text()
    expect(page.locator("#dialog")).not_to_be_visible(timeout=30000)
    expect(page.locator("#content table")).to_contain_text(f"{frames} frames")
    # Read the committed record independently; Chromium can evict an XHR upload response.
    videos = page.request.get(f"{stack.url}/api/projects/{project_id}/videos").json()
    return next(
        video for video in videos if video["group_id"] == group and video["name"] == clip.name
    )


def open_editor(page, http, video_id, *, review=False, keep_page=False):
    with (
        page.expect_popup() as opened,
        page.expect_response(
            lambda r: r.url.endswith(f"/videos/{video_id}/editor") and r.request.method == "POST"
        ) as response,
    ):
        page.get_by_role("button", name="Inspect in CVAT" if review else "CVAT", exact=True).click()
    viewer = opened.value
    navigation = []
    viewer.on("request", lambda r: navigation.append(r.url) if r.is_navigation_request() else None)
    assert response.value.status == 202, response.value.text()
    job_id = response.value.json()["id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = http.get(f"/api/jobs/{job_id}").json()
        assert job["status"] not in {"failed", "cancelled", "interrupted"}, job.get("error")
        if job["status"] == "succeeded":
            result = job["result"]
            target = urljoin(page.url, result["url"])
            navigation_deadline = time.monotonic() + 10
            while target not in navigation and time.monotonic() < navigation_deadline:
                page.wait_for_timeout(50)
            assert target in navigation, "The loading tab must open CVAT without another click"
            if keep_page:
                return result, viewer
            viewer.close()
            return result
        page.wait_for_timeout(200)
    pytest.fail("CVAT task preparation exceeded two minutes")


def login_cvat(page, stack):
    page.goto(stack.cvat_url + "/auth/login")
    page.locator("input#credential").fill(stack.username)
    page.locator("input#password").fill(stack.password)
    page.locator("button[type=submit]").click()
    page.wait_for_url("**/tasks")


def load_job(page, editor):
    page.goto(editor["url"])
    expect(page.locator(".cvat-canvas-container")).to_be_visible(timeout=30000)
    expect(page.locator("#cvat_canvas_background")).to_be_visible(timeout=30000)
    expect(page.locator(".cvat-spinner")).to_have_count(0)


def draw_rectangle(page, label, box, *, kind="Track", source_size=(320, 240)):
    page.locator(".cvat-draw-rectangle-control").click()
    popover = page.locator(".cvat-draw-rectangle-popover")
    expect(popover).to_be_visible()
    popover.locator(".ant-select-selection-item").click()
    page.locator(
        f'.ant-select-dropdown:not(.ant-select-dropdown-hidden) [data-label="{label}"]'
    ).click()
    popover.get_by_role("button", name=kind, exact=True).click()
    # Read only the rendered source-image geometry; all drawing uses mouse events.
    image = page.locator("#cvat_canvas_background").bounding_box()
    assert image and image["width"] > 0 and image["height"] > 0
    for x, y in (box[:2], box[2:]):
        getattr(page, "page", page).mouse.click(
            image["x"] + x * image["width"] / source_size[0],
            image["y"] + y * image["height"] / source_size[1],
        )
    expect(popover).not_to_be_visible()
    item = page.locator(".cvat-objects-sidebar-state-item").last
    expect(item).to_contain_text(label)
    return item


def frame(page, value):
    selector = page.locator(".cvat-player-frame-selector input[role=spinbutton]")
    selector.fill(str(value))
    selector.press("Enter")
    expect(selector).to_have_value(str(value))
    expect(page.locator(".cvat-spinner")).to_have_count(0)


def save_cvat(page):
    with page.expect_response(
        lambda r: (
            re.search(r"/api/jobs/\d+/annotations", r.url) and r.request.method in {"PATCH", "PUT"}
        )
    ) as response:
        page.locator(".cvat-annotation-header-save-button").click()
    assert response.value.ok, response.value.text()
    expect(page.locator(".cvat-spinner")).to_have_count(0)


def submit(page, video_id, editor_id, expected_status=201, *, check_refresh=False):
    page.get_by_role("button", name="Submit saved tracks", exact=True).click()
    page.locator("#action-form select[name=editor]").select_option(editor_id)
    pending = []
    pattern = re.compile(r"/api/projects/[^/]+/videos$")

    def hold(route):
        pending.append(route)

    if check_refresh:
        page.route(pattern, hold)
    try:
        with page.expect_response(
            lambda r: r.url.endswith(f"/videos/{video_id}/cvat-submit")
        ) as response:
            page.get_by_role("button", name="Submit for review", exact=True).click()
        assert response.value.status == expected_status, response.value.text()
        result = response.value.json()
        if check_refresh:
            deadline = time.monotonic() + 10
            while not pending and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert pending, "Submission must refresh the displayed video revision"
            # Hold the real refresh request to reproduce a slow network deterministically.
            expect(page.locator("#dialog")).to_be_visible()
            expect(
                page.get_by_role("button", name="Submit for review", exact=True)
            ).to_be_disabled()
    finally:
        if check_refresh:
            try:
                for route in pending:
                    route.continue_()
            finally:
                page.unroute(pattern, hold)
    if expected_status == 201:
        expect(page.locator("#dialog")).not_to_be_visible(timeout=30000)
    else:
        expect(page.locator("#action-form .form-error")).not_to_be_empty()
    return result


def decide(page, video_id, verdict):
    page.get_by_role("button", name="Review revision", exact=True).click()
    page.locator("#action-form select[name=verdict]").select_option(verdict)
    page.locator("#action-form input[name=comment]").fill("Synthetic E2E review")
    with page.expect_response(lambda r: r.url.endswith(f"/videos/{video_id}/decision")) as response:
        page.get_by_role("button", name="Save decision", exact=True).click()
    assert response.value.status == 200, response.value.text()
    result = response.value.json()
    expect(page.locator("#dialog")).not_to_be_visible()
    return result


def test_video_annotation_review_and_draft_protection(
    video_stack, synthetic_clip, browser_session, video_http, cvat_http
):
    stack = video_stack
    context, errors = browser_session
    page, cvat = context.new_page(), context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Synthetic instrument video E2E")
    video = import_video(page, stack, project_id, synthetic_clip)
    prefix = f"/api/videos/{video['id']}"
    meta = video_http.get(prefix + "/metadata").json()
    assert meta["timestamps"] == pytest.approx([0, 0.1, 0.2, 0.5, 0.6, 0.7])
    assert (video["width"], video["height"], video["frames"]) == (320, 240, 6)
    source = video_http.get(prefix + "/source").content
    assert hashlib.sha256(source).digest() == hashlib.sha256(synthetic_clip.read_bytes()).digest()
    log_step(stack, "browser import preserves original video and variable frame timing")
    editor = open_editor(page, video_http, video["id"])
    login_cvat(cvat, stack)
    load_job(cvat, editor)
    first = draw_rectangle(cvat, "Grasper", [30, 40, 110, 130])
    first_id = first.get_attribute("id")
    draw_rectangle(cvat, "Scissors", [180, 50, 260, 150])
    expect(cvat.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    frame(cvat, 2)
    first = cvat.locator("#" + first_id)
    first.locator(".cvat-object-item-button-occluded").click()
    expect(first.locator(".cvat-object-item-button-occluded-enabled")).to_be_visible()
    # Define reappearance first: CVAT hides outside tracks from the objects sidebar.
    frame(cvat, 5)
    first.locator(".cvat-object-item-button-occluded").click()
    frame(cvat, 4)
    first.locator(".cvat-object-item-button-outside").click()
    frame(cvat, 5)
    expect(first.locator(".cvat-object-item-button-outside-enabled")).to_have_count(0)
    expect(first.locator(".cvat-object-item-button-occluded-enabled")).to_have_count(0)
    save_cvat(cvat)
    page.screenshot(path=stack.artifacts / "imported.png", full_page=True)
    cvat.screenshot(path=stack.artifacts / "native-cvat-tracks.png", full_page=True)
    log_step(stack, "native CVAT drawing and saved occluded/outside keyframes")
    task_id = int(editor["url"].split("/tasks/")[1].split("/")[0])
    before = cvat_http.get(f"/api/tasks/{task_id}/annotations").json()
    assert len(before["tracks"]) == 2 and not before["shapes"]
    resumed = open_editor(page, video_http, video["id"])
    assert resumed == editor
    assert cvat_http.get(f"/api/tasks/{task_id}/annotations").json() == before
    # Reload CVAT itself to prove the draft was saved remotely, not just painted locally.
    load_job(cvat, resumed)
    expect(cvat.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    annotation = submit(page, video["id"], editor["editor_id"])
    saved = video_http.get(prefix + "/tracks").json()
    assert saved["base_revision"] == 1
    tracks = saved["document"]["tracks"]
    assert {t["label_id"] for t in tracks} == {1, 2}
    grasper = next(t for t in tracks if t["label_id"] == 1)
    assert [key["frame"] for key in grasper["keyframes"]] == [0, 2, 4, 5]
    assert grasper["keyframes"][0]["box"] == pytest.approx([30, 40, 110, 130], abs=1)
    assert grasper["keyframes"][1]["occluded"]
    assert grasper["keyframes"][2]["outside"]
    assert not grasper["keyframes"][3]["outside"] and not grasper["keyframes"][3]["occluded"]
    log_step(
        stack, "draft resume and immutable submission preserve both track identities", revision=1
    )
    page.goto(f"{stack.url}/reviews?project={project_id}")
    expect(page.locator(".video-list")).to_contain_text("Pending review")
    review = open_editor(page, video_http, video["id"], review=True)
    assert review["editor_id"] != editor["editor_id"]
    load_job(cvat, review)
    expect(cvat.locator(".cvat-objects-sidebar-state-item")).to_have_count(2)
    frame(cvat, 2)
    # Correct the review copy using the native editor.
    review_first = cvat.locator(".cvat-objects-sidebar-state-item").filter(has_text="Grasper")
    review_first.locator(".cvat-object-item-button-occluded").click()
    save_cvat(cvat)
    assert video_http.get(prefix + "/tracks").json() == saved
    assert decide(page, video["id"], "changes_requested")["revision"] == 1
    log_step(stack, "separate review draft does not change the submitted revision")
    page.locator("#review-filter").select_option("changes_requested")
    # Keep a stale submission dialog open while another task publishes a revision.
    page.get_by_role("button", name="Submit saved tracks", exact=True).click()
    page.locator("#action-form select[name=editor]").select_option(review["editor_id"])
    other = context.new_page()
    other.goto(f"{stack.url}/datasets?project={project_id}")
    current = open_editor(other, video_http, video["id"])
    assert current["editor_id"] != review["editor_id"]
    newer = submit(other, video["id"], current["editor_id"])
    assert newer["revision"] == 2
    assert video_http.get(prefix + "/tracks").json()["document"] == saved["document"]
    review_task = int(review["url"].split("/tasks/")[1].split("/")[0])
    draft = cvat_http.get(f"/api/tasks/{review_task}/annotations").json()
    with page.expect_response(lambda r: r.url.endswith(prefix + "/cvat-submit")) as rejected:
        page.get_by_role("button", name="Submit for review", exact=True).click()
    assert rejected.value.status == 409
    expect(page.locator("#action-form .form-error")).to_contain_text("draft is preserved")
    assert cvat_http.get(f"/api/tasks/{review_task}/annotations").json() == draft
    assert video_http.get(prefix + "/tracks").json()["base_revision"] == 2
    page.screenshot(path=stack.artifacts / "stale-draft-preserved.png", full_page=True)
    page.locator("#close-dialog").click()
    log_step(stack, "stale browser submission rejects without overwriting the CVAT draft")
    page.goto(f"{stack.url}/reviews?project={project_id}")
    final_editor = open_editor(page, video_http, video["id"], review=True)
    load_job(cvat, final_editor)
    frame(cvat, 2)
    cvat.locator(".cvat-objects-sidebar-state-item").filter(has_text="Grasper").locator(
        ".cvat-object-item-button-occluded"
    ).click()
    save_cvat(cvat)
    corrected = submit(page, video["id"], final_editor["editor_id"], check_refresh=True)
    assert corrected["revision"] == 3 and corrected["parent_id"] == newer["id"]
    final = video_http.get(prefix + "/tracks").json()
    assert {t["id"] for t in final["document"]["tracks"]} == {t["id"] for t in tracks}
    assert not next(t for t in final["document"]["tracks"] if t["label_id"] == 1)["keyframes"][1][
        "occluded"
    ]
    assert decide(page, video["id"], "accepted")["revision"] == 3
    versions = video_http.get(prefix + "/revisions").json()
    assert [v["revision"] for v in versions] == [1, 2, 3]
    assert versions[0]["id"] == annotation["id"]
    assert video_http.get(f"/api/projects/{project_id}/assets").json() == []
    log_step(
        stack, "corrected submission and acceptance retain lineage and track identities", revision=3
    )
    # The disk workspace survives an actual server restart; never run two owners concurrently.
    stack.stop_workspace()
    stack.start_workspace()
    page.goto(f"{stack.url}/reviews?project={project_id}")
    page.locator("#review-filter").select_option("accepted")
    expect(page.locator(".video-list")).to_contain_text("Revision 3")
    log_step(stack, "accepted video revision survives workspace restart")
    page.goto(f"{stack.url}/datasets?project={project_id}")
    page.get_by_role("button", name=f"Actions for {synthetic_clip.name}", exact=True).click()
    page.get_by_role("button", name="Delete file", exact=True).click()
    page.locator("#action-form button[type=submit]").click()
    expect(page.locator("#dialog")).not_to_be_visible()
    expect(page.locator("#content")).not_to_contain_text(synthetic_clip.name)
    assert cvat_http.get(f"/api/tasks/{task_id}/annotations").json() == before
    assert cvat_http.get(f"/api/tasks/{review_task}/annotations").json() == draft
    log_step(stack, "browser deletion retains external CVAT tasks and drafts")
    assert errors == []


def test_unsupported_native_shapes_preserve_the_saved_draft(
    video_stack, synthetic_clip, browser_session, video_http, cvat_http
):
    stack = video_stack
    context, errors = browser_session
    page, cvat = context.new_page(), context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Unsupported video shape E2E")
    video = import_video(page, stack, project_id, synthetic_clip)
    editor = open_editor(page, video_http, video["id"])
    login_cvat(cvat, stack)
    load_job(cvat, editor)
    draw_rectangle(cvat, "Grasper", [30, 40, 110, 130], kind="Shape")
    save_cvat(cvat)
    task = int(editor["url"].split("/tasks/")[1].split("/")[0])
    before = cvat_http.get(f"/api/tasks/{task}/annotations").json()
    assert len(before["shapes"]) == 1
    rejected = submit(page, video["id"], editor["editor_id"], expected_status=422)
    assert "draft has been kept" in rejected["detail"]
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json()["base_revision"] == 0
    assert cvat_http.get(f"/api/tasks/{task}/annotations").json() == before
    assert errors == []
    log_step(stack, "unsupported native CVAT shapes cannot silently lose data during submission")


def test_video_evaluation_reservation_and_browser_roles(
    video_stack, synthetic_clip, browser_session, video_http
):
    import io

    from PIL import Image

    stack = video_stack
    context, errors = browser_session
    page = context.new_page()
    login_workspace(page, stack)
    pid = create_project(page, "Video access and held-out grouping E2E")
    video = import_video(
        page, stack, pid, synthetic_clip, group="held-out-procedure", split="validation"
    )
    expect(page.locator("#content table")).to_contain_text("Evaluation only")
    image = io.BytesIO()
    Image.new("RGB", (320, 240)).save(image, format="PNG")
    rejected = video_http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": "frame.png", "group_id": "held-out-procedure", "split": "train"},
        content=image.getvalue(),
    )
    assert rejected.status_code == 409
    imported = video_http.post(
        f"/api/projects/{pid}/assets/upload",
        params={"name": "frame.png", "group_id": "held-out-procedure"},
        content=image.getvalue(),
    )
    assert imported.status_code == 201 and imported.json()["split"] == "validation"
    log_step(stack, "evaluation-only video procedure excludes related image imports from training")
    member = video_http.post(
        "/api/auth/users", json={"username": "video-annotator", "password": stack.password}
    )
    assert member.status_code == 201
    assigned = video_http.put(
        f"/api/projects/{pid}/members",
        json={"user_id": member.json()["id"], "roles": ["annotator"]},
    )
    assert assigned.status_code == 200
    unrelated = video_http.post("/api/projects", json={"name": "Unrelated private project"}).json()[
        "id"
    ]
    context.clear_cookies()
    page.goto(stack.url)
    page.locator("#login-form input[name=username]").fill("video-annotator")
    page.locator("#login-form input[name=password]").fill(stack.password)
    page.locator("#login-button").click()
    expect(page.locator("#workspace")).to_be_visible()
    page.goto(f"{stack.url}/datasets?project={pid}")
    expect(page.locator("#content table")).to_contain_text(synthetic_clip.name)
    expect(page.get_by_role("button", name="CVAT", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Import video", exact=True)).to_have_count(0)
    expect(page.locator("[data-file-selection]")).to_have_count(0)
    assert page.request.get(f"{stack.url}/api/videos/{video['id']}/source").status == 200
    assert page.request.get(f"{stack.url}/api/projects/{unrelated}/videos").status == 403
    assert (
        page.request.post(
            f"{stack.url}/api/videos/{video['id']}/decision",
            data={"base_revision": 0, "verdict": "accepted"},
        ).status
        == 403
    )
    assert (
        page.request.delete(
            f"{stack.url}/api/projects/{pid}/assets", data={"asset_ids": [video["id"]]}
        ).status
        == 403
    )
    assert (
        page.request.post(
            f"{stack.url}/api/projects/{pid}/videos/upload?name=x.mp4&group_id=x", data=b"invalid"
        ).status
        == 403
    )
    assert errors == []
    log_step(
        stack, "browser capabilities and authenticated API enforce annotator/project boundaries"
    )


def test_tool_tracking_sample_from_catalog(video_stack, browser_session, video_http, cvat_http):
    stack = video_stack
    context, errors = browser_session
    page, cvat = context.new_page(), context.new_page()
    login_workspace(page, stack)
    project_id = create_project(page, "Tool tracking sample E2E")
    page.goto(f"{stack.url}/datasets?project={project_id}")

    def import_sample():
        page.get_by_role("button", name="Sample datasets", exact=True).click()
        selector = page.locator('#action-form select[name="template_id"]')
        expect(selector.locator('optgroup[label="Video"] option')).to_have_count(1)
        selector.select_option("hyperkvasir-tool-tracking")
        expect(page.locator('#action-form select[name="split"]')).to_be_disabled()
        expect(page.locator('#action-form [name="include_masks"]')).to_contain_text("Video clip")
        expect(page.locator('#action-form [name="limit"]')).to_have_count(0)
        # Switching back to image datasets restores their controls.
        selector.select_option("Task09_Spleen")
        expect(page.locator('#action-form select[name="split"]')).to_be_enabled()
        expect(page.locator('#action-form [name="include_masks"]')).to_contain_text(
            "Images + labels"
        )
        selector.select_option("hyperkvasir-tool-tracking")
        expect(page.locator("#dataset-import-summary")).to_contain_text("one clip")
        with page.expect_response(lambda r: r.url.endswith("/dataset-imports")) as started:
            page.get_by_role("button", name="Import samples", exact=True).click()
        assert started.value.status == 202
        job_id = started.value.json()["id"]
        expect(page.locator("#dialog")).not_to_be_visible()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = video_http.get(f"/api/jobs/{job_id}").json()
            assert job["status"] not in {"failed", "cancelled", "interrupted"}, job.get("error")
            if job["status"] == "succeeded":
                break
            page.wait_for_timeout(200)
        else:
            pytest.fail("Public video sample import exceeded three minutes")
        expect(page.locator("#messages")).to_contain_text("Video import finished", timeout=10000)
        page.goto(f"{stack.url}/datasets?project={project_id}")
        return job["result"]

    result = import_sample()
    assert result["asset_ids"] == [] and result["annotation_ids"] == []
    videos = video_http.get(f"/api/projects/{project_id}/videos").json()
    assert len(videos) == 1
    video = videos[0]
    assert result["video_ids"] == [video["id"]]
    assert (video["width"], video["height"], video["frames"]) == (720, 576, 1782)
    assert video["duration"] == pytest.approx(71.28)
    original = video_http.get(f"/api/videos/{video['id']}/source").content
    assert hashlib.sha256(original).hexdigest() == (
        "5bfdc1b263785eadf1c6286f63c241b15337ad8e6e589b2c7dbc03026f638164"
    )
    editor = open_editor(page, video_http, video["id"])
    login_cvat(cvat, stack)
    load_job(cvat, editor)
    draw_rectangle(cvat, "Snare", [420, 390, 600, 570], source_size=(720, 576))
    frame(cvat, 25)
    item = cvat.locator(".cvat-objects-sidebar-state-item")
    item.locator(".cvat-object-item-button-keyframe").click()
    save_cvat(cvat)
    cvat.screenshot(path=stack.artifacts / "public-tool-tracking-sample.png", full_page=True)
    annotation = submit(page, video["id"], editor["editor_id"])
    assert annotation["revision"] == 1
    saved = video_http.get(f"/api/videos/{video['id']}/tracks").json()
    assert [key["frame"] for key in saved["document"]["tracks"][0]["keyframes"]] == [0, 25]
    task = int(editor["url"].split("/tasks/")[1].split("/")[0])
    draft = cvat_http.get(f"/api/tasks/{task}/annotations").json()
    assert import_sample()["video_ids"] == [video["id"]]
    assert video_http.get(f"/api/videos/{video['id']}/tracks").json() == saved
    assert cvat_http.get(f"/api/tasks/{task}/annotations").json() == draft
    assert video_http.get(f"/api/projects/{project_id}/assets").json() == []
    assert errors == []
    log_step(
        stack, "public tool-tracking sample imports, opens in CVAT and preserves repeat imports"
    )
