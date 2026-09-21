"""Workspace chat and imports on an HTTP origin outside localhost."""

import re

import pytest
from conftest import VideoStack

from monailabel.core.chat import ChatMessage, ToolCall

pytestmark = pytest.mark.browser_e2e


def test_project_chat_and_import_on_http(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

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
                # Register all gateway choices through the actual form, without inference.
                for provider_model, name in [
                    ("switchyard/openai/gpt-5.6-sol", "GPT-5.6 Sol"),
                    ("azure/openai/gpt-6-astra", "GPT-6 Astra"),
                    ("azure/anthropic/claude-opus-5", "Claude Opus 5"),
                ]:
                    page.locator('#content [data-action="model"]').click()
                    page.locator('#dialog [data-id="hosted"]').click()
                    page.get_by_label("Model", exact=True).select_option(provider_model)
                    with page.expect_response("**/models") as registration:
                        page.get_by_role("button", name="Connect model", exact=True).click()
                    assert registration.value.status == 201, registration.value.text()
                    model = registration.value.json()
                    assert model["name"] == name
                    assert model["config"]["model"] == provider_model
                    assert model["config"]["token_env"] == "NV_INFERENCE_API_KEY"
                    assert "reasoning_effort" not in model["config"]
                    expect(page.locator("#dialog")).not_to_be_visible()
                for service, provider, url, env in [
                    (
                        "openai",
                        "openai-polygons",
                        "https://api.openai.com/v1/responses",
                        "OPENAI_API_KEY",
                    ),
                    (
                        "anthropic",
                        "anthropic-polygons",
                        "https://api.anthropic.com/v1/messages",
                        "ANTHROPIC_API_KEY",
                    ),
                    (
                        "gemini",
                        "openai-chat-polygons",
                        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                        "GEMINI_API_KEY",
                    ),
                    (
                        "nvidia",
                        "openai-chat-polygons",
                        "https://inference-api.nvidia.com/v1/chat/completions",
                        "NV_INFERENCE_API_KEY",
                    ),
                ]:
                    page.locator('#content [data-action="model"]').click()
                    page.locator('#dialog [data-id="hosted"]').click()
                    page.get_by_label("Model service", exact=True).select_option(service)
                    if service == "nvidia":
                        page.get_by_label("Model", exact=True).select_option("custom")
                    identifier = f"my-custom-{service}-vision-model"
                    page.get_by_label("Model ID from the provider", exact=True).fill(identifier)
                    expect(page.get_by_label("Service URL", exact=True)).to_have_value(url)
                    page.get_by_label("API key source", exact=True).select_option("new")
                    page.get_by_label("API key source", exact=True).select_option("environment")
                    expect(
                        page.get_by_label("Environment variable name", exact=True)
                    ).to_have_value(env)
                    page.get_by_label("API key source", exact=True).select_option("new")
                    key = "disposable-api-key-" + service
                    page.get_by_label("API key", exact=True).fill(key)
                    with page.expect_response("**/models") as registration:
                        page.get_by_role("button", name="Connect model", exact=True).click()
                    assert registration.value.status == 201, registration.value.text()
                    model = registration.value.json()
                    assert model["name"] == identifier and model["config"]["model"] == identifier
                    assert model["provider"] == provider and model["config"]["url"] == url
                    assert model["config"]["credential_id"]
                    assert "token_env" not in model["config"]
                    assert key not in registration.value.text()
                    if service == "gemini":
                        assert model["config"]["max_tokens_field"] == "max_tokens"
                    expect(page.locator("#dialog")).not_to_be_visible()
                assert errors == []
            finally:
                browser.close()
    finally:
        stack.stop_workspace()
