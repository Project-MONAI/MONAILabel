"""Workspace chat and imports on an HTTP origin outside localhost."""

import re
import threading

import httpx
import pytest
from conftest import VideoStack

from monailabel.core.chat import ChatMessage, ToolCall
from monailabel.providers.catalog import SERVICES
from monailabel.providers.catalog.presets import HOSTED_PRESETS

pytestmark = pytest.mark.browser_e2e


def test_project_chat_and_import_on_http(tmp_path, monkeypatch, catalog_http):
    from playwright.sync_api import expect, sync_playwright

    choices = {
        "nvidia": ("switchyard/openai/gpt-5.6-sol", "openai-chat-polygons", "NV_INFERENCE_API_KEY"),
        "openai": ("gpt-6-astra", "openai-polygons", "OPENAI_API_KEY"),
        "anthropic": ("claude-opus-5", "anthropic-polygons", "ANTHROPIC_API_KEY"),
        "gemini": ("gemini-3.8-flash", "openai-chat-polygons", "GEMINI_API_KEY"),
    }
    for _, _, env in choices.values():
        monkeypatch.setenv(env, "disposable-environment-key")
    delay_nvidia = threading.Event()
    nvidia_started = threading.Event()
    release_nvidia = threading.Event()
    seen_keys = []

    def provider(request):
        host = request.url.host
        if host == "inference-api.nvidia.com" and delay_nvidia.is_set():
            nvidia_started.set()
            assert release_nvidia.wait(10)
        seen_keys.append(
            request.headers.get(
                "authorization",
                request.headers.get("x-api-key", request.headers.get("x-goog-api-key")),
            )
        )
        assert request.method == "GET"
        if host == "generativelanguage.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3.8-flash",
                            "displayName": "Gemini 3.8 Flash",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )
        service = {
            "inference-api.nvidia.com": "nvidia",
            "api.anthropic.com": "anthropic",
            "api.openai.com": "openai",
        }[host]
        identifier = choices[service][0]
        return httpx.Response(
            200, json={"data": [{"id": identifier}] + [{"id": f"embedding-{i}"} for i in range(40)]}
        )

    catalog_http(provider)
    hostname = "workspace.test"
    monkeypatch.setenv("MONAILABEL_ALLOWED_HOSTS", f"127.0.0.1,{hostname}")
    stack = VideoStack(tmp_path, tmp_path)
    stack.start_workspace()
    stack.app.state.services.assistants.provider.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="create", name="create_project", arguments={"name": "Pathology"})
            ],
        ),
        ChatMessage(role="assistant", content="Created Pathology."),
    ]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=[f"--host-resolver-rules=MAP {hostname} 127.0.0.1", "--no-proxy-server"]
            )
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(stack.url.replace("127.0.0.1", hostname))
                assert page.evaluate("window.isSecureContext") is False
                assert page.evaluate("typeof crypto.randomUUID") == "undefined"
                page.get_by_label("Username", exact=True).fill(stack.username)
                page.get_by_label("Password", exact=True).fill(stack.password)
                page.locator("#login-button").click()
                expect(page.locator("#workspace")).to_be_visible()
                page.locator("#chat-input").fill('Create a project called "Pathology".')
                with page.expect_response("**/api/assistant") as reply:
                    page.locator('#chat-form [type="submit"]').click()
                assert reply.value.status == 200, reply.value.text()
                request = reply.value.request.post_data_json
                assert re.fullmatch(r"[0-9a-f]{32}", request["request_id"])
                expect(page.locator("#project-select option:checked")).to_have_text("Pathology")
                projects = page.evaluate("async () => (await fetch('/api/projects')).json()")
                assert [project["name"] for project in projects] == ["Pathology"]
                page.locator('#content [data-action="dataset"]').click()
                expect(page.locator("#dialog")).to_be_visible()
                expect(page.locator('#dialog input[name="images"]')).to_be_visible()
                page.locator("#close-dialog").click()
                page.locator('#workspace nav [data-page="models"]').click()
                for service, (identifier, provider_name, env) in choices.items():
                    page.locator('#content [data-action="model"]').click()
                    page.locator('#dialog [data-id="hosted"]').click()
                    page.get_by_label("Model service", exact=True).select_option(service)
                    expect(page.locator("#catalog-status")).to_contain_text("compatible model")
                    options = page.locator('select[name="model"] option')
                    expect(options).to_have_count(1)
                    assert options.first.get_attribute("value") == identifier
                    expect(
                        page.locator("#dialog").get_by_role("button", name="Add model", exact=True)
                    ).to_be_disabled()
                    page.get_by_label("Search models", exact=True).fill("no match")
                    expect(options).to_have_count(0)
                    page.get_by_label("Search models", exact=True).fill(identifier)
                    page.get_by_label("Model", exact=True).select_option(identifier)
                    with page.expect_response("**/model-import") as registration:
                        page.locator("#dialog").get_by_role(
                            "button", name="Add model", exact=True
                        ).click()
                    assert registration.value.status == 201, registration.value.text()
                    model = registration.value.json()
                    assert model["config"]["model"] == identifier
                    assert model["provider"] == provider_name
                    assert model["config"]["token_env"] == env
                    assert "disposable-environment-key" not in registration.value.text()
                    expect(page.locator("#dialog")).not_to_be_visible()

                # A new project key discovers and imports a separate connection.
                page.locator('#content [data-action="model"]').click()
                page.locator('#dialog [data-id="hosted"]').click()
                page.get_by_label("Model service", exact=True).select_option("anthropic")
                page.get_by_label("Model", exact=True).select_option("claude-opus-5")
                expect(page.locator("#catalog-selection")).to_have_text(
                    "Already added to this project."
                )
                expect(
                    page.locator("#dialog").get_by_role("button", name="Add model", exact=True)
                ).to_be_disabled()
                page.get_by_label("API key source", exact=True).select_option("new")
                expect(page.locator("#catalog-results")).not_to_be_visible()
                page.get_by_label("API key", exact=True).fill("disposable-saved-key")
                page.get_by_role("button", name="Load models", exact=True).click()
                page.get_by_label("Model", exact=True).select_option("claude-opus-5")
                expect(page.get_by_label("API key source", exact=True)).to_have_value("saved")
                page.get_by_label("Model name", exact=True).fill("My Claude")
                with page.expect_response("**/model-import") as registration:
                    page.locator("#dialog").get_by_role(
                        "button", name="Add model", exact=True
                    ).click()
                model = registration.value.json()
                assert model["name"] == "My Claude"
                assert model["config"]["credential_id"]
                assert "token_env" not in model["config"]
                assert "disposable-saved-key" not in registration.value.text()
                assert seen_keys.count("disposable-saved-key") == 2
                expect(page.locator("#dialog")).not_to_be_visible()

                # Switching providers while a catalog request is pending discards it.
                delay_nvidia.set()
                page.locator('#content [data-action="model"]').click()
                page.locator('#dialog [data-id="hosted"]').click()
                assert nvidia_started.wait(5)
                page.get_by_label("Model service", exact=True).select_option("gemini")
                expect(page.locator('select[name="model"] option')).to_have_text(
                    ["Gemini 3.8 Flash · Already added"]
                )
                with page.expect_response("**/model-catalog"):
                    release_nvidia.set()
                expect(page.locator('select[name="model"] option')).to_have_text(
                    ["Gemini 3.8 Flash · Already added"]
                )
                page.set_viewport_size({"width": 768, "height": 1024})
                assert page.locator("#dialog").evaluate("el => el.scrollWidth <= el.clientWidth")
                page.screenshot(path=str(tmp_path / "model-import.png"))
                page.locator("#close-dialog").click()

                # Manual connection still supports a service outside the catalog.
                page.locator('#content [data-action="model"]').click()
                page.locator('#dialog [data-id="hosted"]').click()
                page.get_by_role("button", name="Connect another vision API", exact=True).click()
                page.get_by_label("Model name", exact=True).fill("Custom vision")
                page.get_by_label("Model ID from the provider", exact=True).fill("custom-model")
                page.get_by_label("Service URL", exact=True).fill(
                    "https://vision.example/v1/chat/completions"
                )
                with page.expect_response("**/models") as registration:
                    page.get_by_role("button", name="Connect model", exact=True).click()
                assert registration.value.status == 201
                assert registration.value.json()["config"]["model"] == "custom-model"
                expect(page.locator("#dialog")).not_to_be_visible()
                assert errors == []
            finally:
                browser.close()
    finally:
        release_nvidia.set()
        stack.stop_workspace()


def test_predefined_provider_switch_in_browser(tmp_path, monkeypatch, catalog_http):
    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("MONAILABEL_PRELOAD_MODELS", "1")
    for service in SERVICES.values():
        monkeypatch.setenv(service.token_env, "disposable-key")

    def provider(request):
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": ("gateway/provider/" + spec.direct_model)
                        if request.url.host == "inference-api.nvidia.com"
                        else spec.direct_model
                    }
                    for spec in HOSTED_PRESETS
                ]
                + (
                    [
                        {"id": "dynamic-cloud/vision/gpt-6-astra", "mode": "responses"},
                        {"id": "gpt-6-astra", "mode": "responses"},
                    ]
                    if request.url.host == "inference-api.nvidia.com"
                    else []
                )
            },
        )

    catalog_http(provider)
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
                project = page.request.post(
                    stack.url + "/api/projects", data={"name": "Preset providers"}
                ).json()
                other = page.request.post(
                    stack.url + "/api/projects", data={"name": "Other project"}
                ).json()
                page.goto(stack.url + "/models")
                page.locator("#project-select").select_option(project["id"])
                prefix = stack.url + f"/api/projects/{project['id']}"
                models = page.request.get(prefix + "/models").json()
                presets = [m for m in models if m["preset"] in {s.key for s in HOSTED_PRESETS}]
                assert len(presets) == 3
                for model in presets:
                    card = page.locator(f'[data-model-id="{model["id"]}"]')
                    expect(card).to_contain_text("Provider: NVIDIA")
                    expect(card.get_by_role("button", name="Change provider")).to_be_visible()
                astra = next(m for m in presets if m["preset"] == "nvidia-astra")
                page.locator(f'[data-model-id="{astra["id"]}"]').get_by_role(
                    "button", name="Change provider"
                ).click()
                expect(page.locator('#dialog select[name="service"] option')).to_have_count(4)
                page.get_by_label("Model service", exact=True).select_option("openai")
                expect(page.get_by_label("Model name", exact=True)).to_have_value("GPT-6 Astra")
                expect(page.get_by_label("Model name", exact=True)).to_have_attribute(
                    "readonly", ""
                )
                with page.expect_response("**/provider") as changed:
                    page.locator("#dialog").get_by_role(
                        "button", name="Change provider", exact=True
                    ).click()
                assert changed.value.status == 200, changed.value.text()
                selected = changed.value.json()
                assert selected["provider"] == "openai-polygons"
                expect(page.locator("#dialog")).not_to_be_visible()
                card = page.locator(f'[data-model-id="{selected["id"]}"]')
                expect(card).to_contain_text("Provider: OpenAI")
                expect(card).to_contain_text("Selected for this project")
                expect(card.get_by_role("heading")).to_have_text("GPT-6 Astra")
                card.get_by_role("button", name="Change provider", exact=True).click()
                page.get_by_label("Model service", exact=True).select_option("automatic")
                with page.expect_response("**/provider") as restored:
                    page.locator("#dialog").get_by_role(
                        "button", name="Change provider", exact=True
                    ).click()
                assert restored.value.status == 200, restored.value.text()
                restored_model = restored.value.json()
                expect(page.locator("#dialog")).not_to_be_visible()
                expect(page.locator(f'[data-model-id="{restored_model["id"]}"]')).to_contain_text(
                    "Provider: NVIDIA"
                )
                page.locator(f'[data-model-id="{restored_model["id"]}"]').get_by_role(
                    "button", name="Change provider", exact=True
                ).click()
                page.get_by_label("Model service", exact=True).select_option("nvidia")
                expect(page.locator('#dialog select[name="model"] option')).to_have_count(3)
                expect(
                    page.locator('#dialog select[name="model"] option[value="gpt-6-astra"]')
                ).to_have_text("gpt-6-astra")
                expect(page.locator('#dialog select[name="model"]')).to_contain_text(
                    "dynamic-cloud/vision"
                )
                page.get_by_label("Model", exact=True).select_option(
                    "dynamic-cloud/vision/gpt-6-astra"
                )
                with page.expect_response("**/provider") as routed:
                    page.locator("#dialog").get_by_role(
                        "button", name="Change provider", exact=True
                    ).click()
                assert routed.value.status == 200, routed.value.text()
                routed_model = routed.value.json()
                assert routed_model["config"]["model"] == "dynamic-cloud/vision/gpt-6-astra"
                assert routed_model["provider"] == "openai-polygons"
                expect(page.locator("#dialog")).not_to_be_visible()
                expect(page.locator(f'[data-model-id="{routed_model["id"]}"]')).to_contain_text(
                    "NVIDIA · dynamic-cloud/vision"
                )
                others = page.request.get(stack.url + f"/api/projects/{other['id']}/models").json()
                assert (
                    next(m for m in others if m["preset"] == "nvidia-astra")["connection_mode"]
                    == "automatic"
                )
                page.screenshot(path=str(tmp_path / "preset-providers.png"), full_page=True)
                assert errors == []
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
