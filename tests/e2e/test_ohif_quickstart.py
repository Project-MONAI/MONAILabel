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

"""Radiology Quickstart through real OHIF with deterministic annotation providers."""

import gzip
import os
import secrets
import ssl

import httpx
import nibabel as nib
import numpy as np
import pytest
from conftest import ROOT, VideoStack
from test_video_cvat import login_workspace

from monailabel.client.client import Client
from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.core.models import ModelRecord
from monailabel.core.ports import Prediction

pytestmark = pytest.mark.browser_e2e


@pytest.mark.parametrize(
    "hostname,https",
    [(os.environ["MONAILABEL_E2E_HOST"], https) for https in (False, True)]
    if "MONAILABEL_E2E_HOST" in os.environ
    else [("127.0.0.1", False), ("ohif.test", False), ("ohif.test", True)],
)
def test_radiology_ohif_annotate_correct_and_submit(tmp_path, monkeypatch, hostname, https):
    from playwright.sync_api import expect, sync_playwright

    artifacts = ROOT / "test-results" / ("ohif-quickstart-" + secrets.token_hex(6))
    artifacts.mkdir(parents=True, mode=0o700)
    print(f"OHIF Quickstart artifacts: {artifacts}", flush=True)
    # A configured build can be reused; otherwise provisioning stays in this test's cache.
    monkeypatch.setenv("MONAILABEL_TOOLS_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("MONAILABEL_ALLOWED_HOSTS", f"127.0.0.1,ohif.test,{hostname}")
    stack = VideoStack(tmp_path, artifacts, https=https)
    stack.start_workspace()
    try:
        with httpx.Client(
            base_url=stack.url,
            timeout=30,
            verify=ssl.create_default_context(cafile=tmp_path / "workspace/.tls/ca.crt")
            if https
            else True,
        ) as http:
            client = Client(http=http)
            client.post("/api/auth/setup", {"username": stack.username, "password": stack.password})
            project = client.post(
                "/api/projects",
                {
                    "name": "Radiology",
                    "labels": [{"id": 0, "name": "Background"}, {"id": 1, "name": "Spleen"}],
                },
            )
            values = np.zeros((24, 28, 12), dtype=np.int16)
            values[6:18, 7:21, 3:9] = 100
            response = http.post(
                f"/api/projects/{project['id']}/assets/upload",
                params={"name": "Quickstart CT.nii.gz", "group_id": "synthetic-ct"},
                content=gzip.compress(nib.Nifti1Image(values, np.eye(4)).to_bytes()),
            )
            response.raise_for_status()
            asset = response.json()
            service = stack.app.state.services
            calls = []

            class AnnotationFixture:
                def predict(self, image, labels, prompt, model):
                    calls.append((model.name, image.shape))
                    return Prediction((image[..., 0] > 0.5).astype(np.uint8))

            monkeypatch.setattr(service.models.recipes, "segmenter", lambda *_: AnnotationFixture())
            service.models.providers["openai-chat-polygons"] = AnnotationFixture()
            with service.store.transaction() as session:
                for name, provider, config in [
                    ("VISTA3D", "vista3d", {}),
                    (
                        "GPT-6 Astra",
                        "openai-chat-polygons",
                        {"url": "http://unused.test", "model": "fixture"},
                    ),
                ]:
                    session.insert(
                        ModelRecord(
                            project_id=project["id"],
                            name=name,
                            provider=provider,
                            config=config,
                            label_ids=[0, 1],
                            read_only=True,
                        )
                    )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    channel=os.environ.get("MONAILABEL_E2E_BROWSER_CHANNEL"),
                    args=["--host-resolver-rules=MAP ohif.test 127.0.0.1", "--no-proxy-server"],
                )
                context = browser.new_context(
                    viewport={"width": 1600, "height": 1000}, ignore_https_errors=https
                )
                context.add_init_script(path=str(ROOT / "tests/e2e/speech_fixture.js"))
                errors = []
                context.on(
                    "page",
                    lambda page: page.on("pageerror", lambda error: errors.append(str(error))),
                )
                try:
                    stack.url = stack.url.replace("127.0.0.1", hostname)
                    page = context.new_page()
                    login_workspace(page, stack)
                    assert page.evaluate("isSecureContext") == (https or hostname == "127.0.0.1")
                    page.goto(f"{stack.url}/datasets?project={project['id']}")
                    with page.expect_popup() as opened:
                        page.get_by_role("button", name="OHIF", exact=True).click()
                    viewer = opened.value
                    viewer.wait_for_url("**/ohif/**", timeout=1800000)
                    send = viewer.get_by_role("button", name="Send prompt", exact=True)
                    expect(send).to_be_enabled(timeout=120000)
                    microphone = viewer.get_by_role("button", name="Use microphone", exact=True)
                    prompt_box = viewer.get_by_role("textbox", name="Annotation prompt")
                    if https or hostname == "127.0.0.1":
                        expect(microphone).to_be_enabled()
                        prompt_box.fill("Annotate")
                        microphone.click()
                        expect(viewer.get_by_role("button", name="Stop microphone")).to_be_visible()
                        viewer.evaluate("speechFixture.result('the spleen')")
                        expect(prompt_box).to_have_value("Annotate the spleen")
                        viewer.evaluate("speechFixture.result('the spleen on this slice')")
                        expect(prompt_box).to_have_value("Annotate the spleen on this slice")
                        viewer.evaluate("speechFixture.end()")
                        expect(microphone).to_be_visible()
                        microphone.click()
                        viewer.evaluate("speechFixture.error('not-allowed')")
                        expect(viewer.locator("#monailabel-voice-status")).to_contain_text("denied")
                        assert viewer.evaluate("speechFixture.aborted") == 0
                        viewer.screenshot(path=str(artifacts / "ohif-voice.png"))
                        prompt_box.fill("")
                    else:
                        expect(microphone).to_be_disabled()
                        expect(viewer.locator("#monailabel-voice-status")).to_contain_text("HTTPS")
                    canvas = viewer.locator("canvas").first
                    canvas.hover()
                    for _ in range(5):
                        viewer.mouse.wheel(0, 120)
                        viewer.wait_for_timeout(100)

                    def prompt(message, tool=None, arguments=None):
                        if tool:
                            service.assistants.provider.queue.append(
                                ChatMessage(
                                    role="assistant",
                                    tool_calls=[
                                        ToolCall(
                                            id=secrets.token_hex(8), name=tool, arguments=arguments
                                        )
                                    ],
                                )
                            )
                        viewer.get_by_role("textbox", name="Annotation prompt").fill(message)
                        send.click()

                    for count, (message, scope, name) in enumerate(
                        [
                            (
                                "Segment the spleen in the whole volume using VISTA3D.",
                                "full",
                                "VISTA3D",
                            ),
                            (
                                "Annotate the spleen on the current slice using GPT Astra.",
                                "current_slice",
                                "GPT-6 Astra",
                            ),
                        ],
                        start=1,
                    ):
                        if count == 2:
                            prompt(
                                "Clear the spleen annotation on the current slice.",
                                "clear_segments",
                                {"targets": ["spleen"], "scope": "current_slice"},
                            )
                            expect(viewer.get_by_role("log")).to_contain_text(
                                "Cleared the requested annotation", timeout=30000
                            )
                            model_calls = len(service.assistants.provider.calls)
                            prompt("Undo that.")
                            expect(viewer.get_by_role("log")).to_contain_text(
                                "Restored the previous mask.", timeout=30000
                            )
                            prompt("Redo that.")
                            expect(
                                viewer.get_by_role("log").get_by_text(
                                    "Restored the previous mask.", exact=True
                                )
                            ).to_have_count(2, timeout=30000)
                            assert len(service.assistants.provider.calls) == model_calls
                            assert not service.assistants.provider.queue
                        prompt(
                            message,
                            "annotate",
                            {"targets": ["spleen"], "scope": scope, "model_name": name},
                        )
                        expect(
                            viewer.get_by_role("log").get_by_text(
                                "Applied the editable proposal.", exact=False
                            )
                        ).to_have_count(count, timeout=60000)
                        assert client.get(f"/api/assets/{asset['id']}")["revision"] == 0
                    prompt(
                        "Submit this annotation for review.", "viewer_edit", {"operation": "submit"}
                    )
                    expect(viewer.get_by_role("log")).to_contain_text(
                        "Revision 1 saved", timeout=30000
                    )
                    saved = client.get(f"/api/assets/{asset['id']}")
                    mask = http.get(f"/api/annotations/{saved['annotation_id']}/mask.bin").content
                    assert len(mask) == values.size and any(mask)
                    assert calls[0] == ("VISTA3D", (*values.shape, 1))
                    assert calls[1][0] == "GPT-6 Astra" and len(calls[1][1]) == 3
                    assert not errors
                finally:
                    for index, window in enumerate(context.pages):
                        window.screenshot(path=str(artifacts / f"page-{index}.png"))
                        (artifacts / f"page-{index}.html").write_text(window.content())
                    context.close()
                    browser.close()
    finally:
        stack.stop_workspace()
