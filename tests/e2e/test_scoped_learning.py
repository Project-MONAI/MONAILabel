"""Inspect independent review scopes and train from accepted coverage in the browser."""

import io
import subprocess
import time

import numpy as np
import pytest
from conftest import VideoStack
from PIL import Image

pytestmark = pytest.mark.browser_e2e


def seed(page, base, directory, video):
    def post(path, payload):
        response = page.request.post(base + "/api" + path, data=payload)
        assert response.ok, response.text()
        return response.json()

    project = post(
        "/projects",
        {
            "name": "Tool learning" if video else "Nuclei learning",
            "labels": [
                {"id": 0, "name": "Background"},
                {"id": 1, "name": "Snare" if video else "Nuclei"},
            ],
        },
    )
    prefix = f"/projects/{project['id']}"
    if video:
        clip = directory / "instruments.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10",
                "-frames:v",
                "6",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ],
            check=True,
            capture_output=True,
        )
        response = page.request.post(
            base + "/api" + prefix + "/videos/upload",
            params={"name": "instruments.mp4", "group_id": "procedure-1", "labels": "Snare"},
            data=clip.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
    else:
        content = io.BytesIO()
        Image.fromarray(np.full((64, 64, 3), [175, 120, 170], np.uint8)).save(content, "PNG")
        response = page.request.post(
            base + "/api" + prefix + "/assets/upload",
            params={"name": "nuclei.png", "group_id": "slide-1"},
            data=content.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
        )
    assert response.ok, response.text()
    source = response.json()
    if video:
        document = {
            "tracks": [
                {
                    "id": str(start),
                    "label_id": 1,
                    "keyframes": [
                        {"frame": frame, "points": [8, 8, 24, 8, 24, 24, 8, 24]}
                        for frame in range(start, start + 2)
                    ]
                    + [
                        {
                            "frame": start + 2,
                            "points": [8, 8, 24, 8, 24, 24, 8, 24],
                            "outside": True,
                        }
                    ],
                }
                for start in [0, 3]
            ]
        }
        post(f"/videos/{source['id']}/review", {"base_revision": 0, "document": document})
    else:
        mask = np.zeros((64, 64), np.uint8)
        mask[8:16, 8:16] = mask[40:48, 40:48] = 1
        post(
            f"/assets/{source['id']}/review",
            {
                "base_revision": 0,
                "mask": mask.tolist(),
                "covered_labels": [0, 1],
                "regions": [
                    {"x": start, "y": start, "width": 24, "height": 24} for start in [0, 32]
                ],
            },
        )
    return project, source


@pytest.mark.parametrize("video", [False, True], ids=["pathology", "endoscopy"])
def test_review_scopes_train_without_evaluation(tmp_path, video):
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
                project, source = seed(page, stack.url, tmp_path, video)
                prefix = stack.url + f"/api/projects/{project['id']}"
                units = page.request.get(prefix + "/review-units").json()
                assert len(units) == 2
                page.goto(stack.url + "/reviews")
                page.locator("#project-select").select_option(project["id"])
                expect(page.locator('[data-action="review-unit"]')).to_have_count(2)
                for unit, verdict in zip(units, ["accepted", "changes_requested"], strict=True):
                    page.locator(f'[data-action="review-unit"][data-id="{unit["id"]}"]').click()
                    picture = page.locator("[data-review-inspection] img")
                    expect(picture).to_be_visible()
                    page.wait_for_function(
                        "document.querySelector('[data-review-inspection] img').naturalWidth > 0"
                    )
                    if video:
                        slider = page.locator("[data-review-frame]")
                        slider.fill(str(unit["scope"]["stop"] - 1))
                        slider.dispatch_event("change")
                        expect(page.locator("[data-frame-number]")).to_have_text(
                            str(unit["scope"]["stop"] - 1)
                        )
                    page.get_by_label("Show segmentation").uncheck()
                    page.get_by_label("Decision", exact=True).select_option(verdict)
                    page.get_by_role("button", name="Save review", exact=True).click()
                    expect(page.locator("#dialog")).not_to_be_visible()
                decisions = page.request.get(prefix + "/decisions").json()
                assert {d["annotation_id"]: d["verdict"] for d in decisions} == {
                    units[0]["annotation_id"]: "accepted",
                    units[1]["annotation_id"]: "changes_requested",
                }
                page.locator('#workspace nav [data-page="models"]').click()
                page.locator('#content [data-action="model"]').click()
                page.locator('#dialog [data-id="learner"]').click()
                page.get_by_label("Model name", exact=True).fill("Local specialist")
                with page.expect_response("**/learners") as created:
                    page.get_by_role("button", name="Create model", exact=True).click()
                assert created.value.status == 201, created.value.text()
                learner = created.value.json()
                assert learner["config"]["spatial_dims"] == 2
                expect(page.locator("#dialog")).not_to_be_visible()
                page.locator(f'[data-action="start-training"][data-id="{learner["id"]}"]').click()
                expect(
                    page.get_by_role("combobox", name="Evaluation set", exact=True)
                ).to_have_value("none")
                page.get_by_text("Training settings (recommended)", exact=True).click()
                page.get_by_label("Epochs", exact=True).fill("1")
                page.get_by_label("Training steps per epoch", exact=True).fill("2")
                page.get_by_label("Training crop size", exact=True).fill("16")
                page.get_by_role("combobox", name="Run training on", exact=True).select_option(
                    "cpu"
                )
                with page.expect_response("**/train") as started:
                    page.locator("#dialog").get_by_role(
                        "button", name="Start training", exact=True
                    ).click()
                assert started.value.status == 202, started.value.text()
                job = started.value.json()
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    job = page.request.get(stack.url + "/api/jobs/" + job["id"]).json()
                    if job["status"] in {"succeeded", "failed", "cancelled"}:
                        break
                    time.sleep(0.1)
                assert job["status"] == "succeeded", job
                report = page.request.get(
                    stack.url + f"/api/jobs/{job['id']}/training-report"
                ).json()
                assert report["metrics"] is None and report["error"] is None
                snapshots = page.request.get(prefix + "/snapshots")
                assert snapshots.ok, snapshots.text()
                snapshot = next(
                    s for s in snapshots.json() if s["id"] == job["result"]["snapshot_id"]
                )
                assert {s["unit_id"] for s in snapshot["samples"]} == {units[0]["id"]}
                assert all(
                    s["asset_id"] == source["id"] and s["split"] == "train"
                    for s in snapshot["samples"]
                )
                assert len(snapshot["samples"]) == (2 if video else 1)
                assert errors == []
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
