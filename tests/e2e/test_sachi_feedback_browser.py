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

"""Production CVAT controls against deterministic transport, no paid models or user data."""

import json
from pathlib import Path

import pytest
from conftest import VideoStack

pytestmark = pytest.mark.browser_e2e
STATIC = Path(__file__).parents[2] / "packages/server/src/monailabel/server/static"


@pytest.mark.parametrize("review", [False, True])
def test_cvat_completion_choices_defaults_and_errors(tmp_path, review):
    from playwright.sync_api import expect, sync_playwright

    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    state = {"submitted": False, "fail": False, "editor_requests": 0}
    info = {
        "project": {
            "id": "p",
            "annotation_model_id": "sol",
            "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Tool"}],
        },
        "video": {
            "id": "v",
            "name": "Synthetic clip",
            "revision": 0,
            "frames": 25,
            "width": 32,
            "height": 32,
            "annotation_id": "annotation" if review else None,
        },
        "roles": ["annotator", "reviewer"],
        "editor": {
            "mode": "review" if review else "annotation",
            "base_revision": 0,
            "task_id": 1,
            "job_id": 1,
            "label_map": {"1": 1},
        },
        "detection_models": [
            {"id": "sol", "name": "GPT Sol", "preset": "demo-gpt-sol"},
            {"id": "astra", "name": "GPT Astra", "preset": "demo-gpt-astra"},
        ],
    }

    def route(request):
        path = request.request.url.split(stack.url)[-1]
        result = {}
        code = 200
        if path.startswith("/api/cvat/editors/"):
            result = json.loads(json.dumps(info))
            result["editor"]["submitted_annotation_id"] = (
                "annotation" if state["submitted"] and path.endswith("/first") else None
            )
            result["video"]["revision"] = int(state["submitted"])
        elif path.endswith("/cvat-submit"):
            if state["fail"]:
                code, result = 409, {"detail": "The revision changed. Your draft is preserved."}
            else:
                state["submitted"] = True
                result = {"revision": 1}
        elif path.endswith("/decision"):
            result = {"verdict": request.request.post_data_json["verdict"]}
        elif path == "/api/videos/v/editor":
            state["editor_requests"] += 1
            assert request.request.post_data_json["base_revision"] == 1
            result = {"id": "prepare"}
        elif path == "/api/jobs/prepare":
            result = {"status": "succeeded", "result": {"url": "/cvat/editor/next"}}
        request.fulfill(status=code, json=result)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/**", route)
            page.route(
                "**/cvat/editor/*",
                lambda r: r.fulfill(
                    content_type="text/html", body=(STATIC / "cvat.html").read_text()
                ),
            )
            page.route(
                "**/tasks/*/jobs/*",
                lambda r: r.fulfill(
                    content_type="text/html",
                    body="""
                <script>window.monaiVideo = {
                    ready: () => true, frame: () => 0, hasSelection: () => false,
                    objects: () => [], configureHints: async () => {},
                    save: async () => { window.saved = (window.saved || 0) + 1; }
                };</script><p>Synthetic native editor</p>""",
                ),
            )
            page.goto(stack.url + "/cvat/editor/first")
            expect(page.locator("#detection-model")).to_have_value("astra")
            expect(page.locator("#prompt")).to_have_attribute(
                "placeholder",
                "Locate and segment the tool using GPT Astra. Track it for the next 20 frames.",
            )
            page.locator("#detection-model").select_option("sol")
            expect(page.locator("#detection-model")).to_have_value("sol")
            if review:
                page.locator("#accept").click()
                expect(page.get_by_role("dialog")).to_be_visible()
                expect(page.locator("#completion-title")).to_have_text("Review submitted")
                page.get_by_role("button", name="Keep working", exact=True).click()
                expect(page.get_by_role("dialog")).not_to_be_visible()
                page.locator("#changes").click()
                expect(page.locator("#completion-description")).to_contain_text("Needs changes")
            else:
                state["fail"] = True
                page.locator("#submit").click()
                expect(page.locator("#status")).to_contain_text("draft is preserved")
                expect(page.get_by_role("dialog")).not_to_be_visible()
                state["fail"] = False
                page.locator("#submit").click()
                expect(page.locator("#completion-title")).to_have_text("Annotation submitted")
                page.screenshot(path=tmp_path / "cvat-submitted.png")
                page.get_by_role("button", name="Keep working", exact=True).click()
                page.wait_for_url("**/cvat/editor/next")
                assert state["editor_requests"] == 1
                expect(page.locator("#submit")).to_be_enabled()
                page.locator("#submit").click()
                expect(page.get_by_role("dialog")).to_be_visible()
            page.get_by_role("button", name="Return to workspace", exact=True).click()
            page.wait_for_url("**/" + ("reviews" if review else "datasets") + "?project=p")
            assert errors == []
            browser.close()
    finally:
        stack.stop_workspace()
