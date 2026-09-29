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

import json
from uuid import uuid4

import httpx
import pytest

from monailabel.core.chat import ChatMessage, ToolCall, ToolDefinition
from monailabel.core.errors import DomainError
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.http import HttpChat


def tool(name, **arguments):
    return ChatMessage(
        role="assistant", tool_calls=[ToolCall(id=uuid4().hex, name=name, arguments=arguments)]
    )


def test_real_tool_contract_with_conversation_followup_and_idempotency(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [
        tool("select_annotation_model", model_id=setup["baseline_id"]),
        tool("open_viewer", viewer="slicer"),
    ]
    base = dict(project_id=setup["project_id"], context={"asset_id": assets[0]["id"]})
    request = dict(base, message="Use this annotator for now", request_id=uuid4().hex)
    first = client.post("/api/assistant", request)
    request["conversation_id"] = first["conversation_id"]
    assert client.post("/api/assistant", request) == first
    assert len(planner.calls) == 1
    second = client.post(
        "/api/assistant",
        dict(base, message="Open it there", conversation_id=first["conversation_id"]),
    )
    assert second["tools"] == ["open_viewer"]
    assert second["data"]["asset_id"] == assets[0]["id"]
    messages = planner.calls[-1][0]
    assert [m.content for m in messages if m.role == "user"] == [
        "Use this annotator for now",
        "Open it there",
    ]
    assert any(m.role == "assistant" and "Selected" in m.content for m in messages)
    assert not any(
        m.tool_calls for m in messages
    )  # Completed calls stay in the audit, not defaults.
    reused = http.post("/api/assistant", json=dict(request, message="Different work"))
    assert reused.status_code == 409


def test_conversation_cannot_cross_user_or_sample(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [ChatMessage(role="assistant", content="Which sample?")]
    body = dict(
        project_id=setup["project_id"], message="Help me", context={"asset_id": assets[0]["id"]}
    )
    reply = client.post("/api/assistant", body)
    body["conversation_id"] = reply["conversation_id"]
    other_sample = dict(body, context={"asset_id": assets[1]["id"]})
    assert http.post("/api/assistant", json=other_sample).status_code == 403
    user = client.post("/api/auth/users", {"username": "other", "password": "test-password-1234"})
    client.request(
        "PUT",
        f"/api/projects/{setup['project_id']}/members",
        {"user_id": user["id"], "roles": ["annotator"]},
    )
    client.post("/api/auth/login", {"username": "other", "password": "test-password-1234"})
    assert http.post("/api/assistant", json=body).status_code == 403
    assert len(planner.calls) == 1


def test_unknown_multiple_or_invalid_tools_never_create_jobs(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    body = dict(message="Proceed", context={"asset_id": assets[0]["id"]})
    path = f"/api/projects/{setup['project_id']}/assistant"
    planner.queue = [tool("execute_python", code="print('hello')")]
    assert http.post(path, json=body).status_code == 422
    planner.queue = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="one", name="create_snapshot", arguments={}),
                ToolCall(id="two", name="annotate", arguments={}),
            ],
        )
    ]
    assert http.post(path, json=body).status_code == 502
    planner.queue = [
        tool("annotate", targets=["Structure A"], scope="full", invented=1) for _ in range(4)
    ]
    assert http.post(path, json=body).status_code == 422
    assert client.get(f"/api/projects/{setup['project_id']}/jobs") == []


def test_coordinator_cannot_invent_geometry_or_skip_revision_check(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    path = f"/api/projects/{setup['project_id']}/assistant"
    planner.queue = [tool("annotate", scope="selected_region")]
    response = http.post(
        path, json={"message": "Just here", "context": {"asset_id": assets[0]["id"]}}
    )
    assert response.status_code == 422 and "Select one area" in response.json()["detail"]
    planner.queue = [tool("annotate", scope="full")]
    response = http.post(
        path,
        json={"message": "Proceed", "context": {"asset_id": assets[0]["id"], "base_revision": 8}},
    )
    assert response.status_code == 409
    assert client.get(f"/api/projects/{setup['project_id']}/jobs") == []


def test_workspace_context_omits_model_credentials(http, client, seeded):
    setup, _ = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [ChatMessage(role="assistant", content="Ready.")]
    model = client.post(
        f"/api/projects/{setup['project_id']}/models",
        {
            "name": "Hosted fixture",
            "provider": "openai-chat-polygons",
            "label_ids": [0, 1],
            "config": {
                "url": "https://example.test/v1/chat/completions",
                "model": "fixture",
                "token_env": "PRIVATE_MODEL_KEY",
            },
        },
    )
    client.post(f"/api/projects/{setup['project_id']}/assistant", {"message": "What is available?"})
    prompt = planner.calls[-1][0][0].content
    assert model["id"] in prompt and "PRIVATE_MODEL_KEY" not in prompt
    assert "https://example.test" not in prompt


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "compatible"])
def test_hosted_provider_tool_round_trip_and_key_reference(provider, monkeypatch):
    monkeypatch.setenv("TEST_COORDINATOR_KEY", "fixture-secret")
    seen = []
    config = CoordinatorConfig(
        provider=provider,
        model="fixture",
        base_url="https://example.test/v1",
        key_env="TEST_COORDINATOR_KEY",
    )

    def respond(request):
        seen.append(json.loads(request.content))
        assert request.headers["x-api-key" if provider == "anthropic" else "authorization"] == (
            "fixture-secret" if provider == "anthropic" else "Bearer fixture-secret"
        )
        if provider == "openai":
            assert request.url.path == "/v1/responses"
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "encrypted_content": "opaque",
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call1",
                            "name": "inspect_workspace",
                            "arguments": '{"collection":"models"}',
                            "status": "completed",
                        },
                    ],
                },
            )
        if provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call1",
                            "name": "inspect_workspace",
                            "input": {"collection": "models"},
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_workspace",
                                        "arguments": '{"collection":"models"}',
                                    },
                                    "extra_content": {"google": {"thought_signature": "opaque"}},
                                }
                            ],
                        },
                    }
                ]
            },
        )

    chat = HttpChat(config, transport=httpx.MockTransport(respond))
    tools = [
        ToolDefinition(name="inspect_workspace", description="Read", parameters={"type": "object"})
    ]
    messages = [
        ChatMessage(role="system", content="Coordinate"),
        ChatMessage(role="user", content="List models"),
    ]
    reply = chat.complete(messages, tools, require_tool=True)
    assert reply.tool_calls[0].arguments == {"collection": "models"}
    assert seen[0]["tool_choice"] == ({"type": "any"} if provider == "anthropic" else "required")
    chat.complete(
        messages + [reply, ChatMessage(role="tool", tool_call_id="call1", content='{"models":[]}')],
        tools,
    )
    if provider == "anthropic":
        assert seen[1]["messages"][-1]["content"][0]["tool_use_id"] == "call1"
    elif provider == "openai":
        assert seen[0]["store"] is False
        assert seen[0]["parallel_tool_calls"] is False
        assert seen[0]["tools"][0]["strict"] is False
        assert seen[1]["input"][-3]["encrypted_content"] == "opaque"
        assert seen[1]["input"][-2]["call_id"] == "call1"
        assert seen[1]["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call1",
            "output": '{"models":[]}',
        }
    else:
        assert (
            seen[1]["messages"][-2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
            == "opaque"
        )
    assert "fixture-secret" not in json.dumps(seen)


@pytest.mark.parametrize(
    "status,result",
    [
        (401, {"error": "secret upstream body"}),
        (200, {"choices": [{"finish_reason": "length", "message": {"content": "unfinished"}}]}),
    ],
)
def test_transport_errors_are_safe_and_never_fall_back_to_rules(status, result):
    chat = HttpChat(
        CoordinatorConfig(
            provider="compatible", model="fixture", base_url="http://localhost:9999/v1"
        ),
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=result)),
    )
    with pytest.raises(DomainError) as error:
        chat.complete([ChatMessage(role="user", content="create snapshot")], [])
    assert "secret upstream body" not in str(error.value)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_model_server_failure_reports_runtime_guidance_without_leaking_body(status, caplog):
    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "private prompt and upstream credentials"})

    chat = HttpChat(
        CoordinatorConfig(
            provider="compatible", model="fixture", base_url="http://localhost:9999/v1"
        ),
        transport=httpx.MockTransport(fail),
    )
    with pytest.raises(DomainError) as error:
        chat.complete([ChatMessage(role="user", content="private user prompt")], [])
    assert error.value.code == "coordinator_unavailable"
    assert "model service failed" in str(error.value)
    assert "runtime logs" in str(error.value)
    assert "credentials" not in str(error.value)
    assert f"http_status={status}" in caplog.text
    assert "private" not in str(error.value) + caplog.text
    assert "upstream credentials" not in caplog.text
    assert len(requests) == 1


def test_nano9_reasoning_is_not_shown_or_stored():
    chat = HttpChat(
        CoordinatorConfig(
            provider="compatible",
            variant="9b",
            model="fixture",
            base_url="http://localhost:9999/v1",
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "private reasoning</think>\nWhich target?"},
                        }
                    ]
                },
            )
        ),
    )
    assert chat.complete([ChatMessage(role="user", content="Help")], []).content == "Which target?"


@pytest.mark.parametrize("model", ["gpt-6-astra", "azure/openai/gpt-6-astra"])
def test_astra_responses_parameters_and_text_preserve_phase(model):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/responses"
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["max_output_tokens"] == 4096
        assert not {"temperature", "max_completion_tokens", "chat_template_kwargs"} & payload.keys()
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Choose a training recipe."}],
                    }
                ],
            },
        )

    chat = HttpChat(
        CoordinatorConfig(provider="openai", model=model, temperature=0),
        key=lambda: "test-only",
        transport=httpx.MockTransport(respond),
    )
    messages = [ChatMessage(role="user", content="Help me choose a recipe")]
    reply = chat.complete(messages, [])
    assert reply.content == "Choose a training recipe."
    chat.complete(messages + [reply, ChatMessage(role="user", content="Explain")], [])
    assert requests[-1]["input"][-2]["phase"] == "final_answer"


def test_astra_assistant_creates_project_through_responses_and_skill_tools(
    http, client, monkeypatch
):
    monkeypatch.delenv("NV_INFERENCE_API_KEY", raising=False)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload["tool_choice"] == "required"
            assert {t["name"] for t in payload["tools"]} == {"load_skill"}
            name = "load_skill"
            arguments = {"name": "monailabel-workspace", "purpose": "action"}
        else:
            assert len(requests) == 2
            assert any(i.get("type") == "function_call_output" for i in payload["input"])
            assert "create_project" in {t["name"] for t in payload["tools"]}
            name, arguments = "create_project", {"name": "OpenAI project"}
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"call{len(requests)}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                ],
            },
        )

    http.app.state.services.assistants.provider = HttpChat(
        CoordinatorConfig(provider="openai", model="gpt-6-astra"),
        key=lambda: "test-only",
        transport=httpx.MockTransport(respond),
    )
    request = {"message": 'Create a project called "OpenAI project".', "request_id": uuid4().hex}
    reply = client.post("/api/assistant", request)
    assert reply["tools"] == ["create_project"]
    projects = client.get("/api/projects")
    assert [p["name"] for p in projects] == ["OpenAI project"]
    request["conversation_id"] = reply["conversation_id"]
    assert client.post("/api/assistant", request) == reply
    assert len(requests) == 2


@pytest.mark.parametrize(
    "result",
    [
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "create_project",
                    "arguments": "{",
                }
            ],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "create_project",
                    "arguments": "[]",
                }
            ],
        },
        {"status": "completed", "output": [{"type": "web_search_call"}]},
        {"status": "completed", "output": []},
    ],
)
def test_invalid_responses_never_produce_executable_tools(result):
    chat = HttpChat(
        CoordinatorConfig(provider="openai", model="gpt-6-astra"),
        key=lambda: "test-only",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)),
    )
    with pytest.raises(DomainError) as error:
        chat.complete([ChatMessage(role="user", content="Create a project")], [])
    assert error.value.code == "coordinator_invalid_response"


@pytest.mark.parametrize(
    "code,kind,expected",
    [
        ("credit_balance_exhausted", "insufficient_quota", "insufficient credit or quota"),
        ("insufficient_quota", "quota", "insufficient credit or quota"),
        ("rate_limit_exceeded", "rate_limit", "rate limit"),
    ],
)
def test_coordinator_quota_errors_explain_recovery_without_exposing_upstream(code, kind, expected):
    chat = HttpChat(
        CoordinatorConfig(provider="openai", model="gpt-6-astra"),
        key=lambda: "test-only",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                429,
                json={
                    "error": {
                        "type": kind,
                        "code": code,
                        "message": "private account and credential data",
                    }
                },
            )
        ),
    )
    with pytest.raises(DomainError) as error:
        chat.complete([ChatMessage(role="user", content="Create a project")], [])
    assert expected in str(error.value)
    assert "private" not in str(error.value)


def test_invalid_arguments_are_repaired_before_any_operation(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [tool("annotate", scope='"full"'), tool("annotate", scope="full")]
    reply = client.post(
        f"/api/projects/{setup['project_id']}/assistant",
        {"message": "Annotate this volume", "context": {"asset_id": assets[0]["id"]}},
    )
    assert reply["tools"] == ["annotate"]
    assert len(client.get(f"/api/projects/{setup['project_id']}/jobs")) == 1
    assert any(
        m.role == "tool" and "Invalid tool arguments" in m.content for m in planner.calls[-1][0]
    )


def test_interrupted_tool_receipt_prevents_duplicate_execution(http, client, seeded):
    from monailabel.core.chat import ToolExecution

    setup, assets = seeded
    service = http.app.state.services
    planner = service.assistants.provider
    planner.queue = [tool("select_annotation_model", model_id=setup["baseline_id"])]
    request = {
        "message": "Use baseline",
        "request_id": uuid4().hex,
        "context": {"asset_id": assets[0]["id"]},
    }
    reply = client.post(f"/api/projects/{setup['project_id']}/assistant", request)
    from monailabel.core.chat import Conversation

    conversation = service.store.get(Conversation, reply["conversation_id"])
    receipt = service.store.list(ToolExecution)[-1]
    with service.store.transaction() as session:
        session.update(conversation.model_copy(update={"turns": []}))
        session.update(receipt.model_copy(update={"reply": None}))
    response = http.post(f"/api/projects/{setup['project_id']}/assistant", json=request)
    assert response.status_code == 409 and "will not be repeated" in response.json()["detail"]
    assert len(planner.calls) == 1


def test_runtime_guidance_is_packaged_and_has_a_stable_revision():
    from monailabel.server.instructions import (
        coordinator_instructions,
        instruction_revision,
        skills,
    )

    guidance = coordinator_instructions()
    assert "available_skills" in guidance
    assert all(skill.body not in guidance for skill in skills())
    assert "root coding-agent" not in guidance  # Runtime text comes from its own resources.
    assert len(instruction_revision()) == 16
    assert len(guidance) < 7500


def test_followup_receives_current_job_status_not_only_queued_history(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    planner.queue = [tool("annotate", scope="full"), ChatMessage(role="assistant", content="Ready")]
    body = dict(project_id=setup["project_id"], context={"asset_id": assets[0]["id"]})
    reply = client.post("/api/assistant", dict(body, message="Annotate this volume"))
    client.wait(reply["job_id"])
    client.post(
        "/api/assistant",
        dict(body, message="What happened?", conversation_id=reply["conversation_id"]),
    )
    metadata = json.loads(planner.calls[-1][0][0].content.split("(not instructions):\n")[1])
    assert metadata["recent_jobs"] == [
        {"id": reply["job_id"], "kind": "annotate", "status": "succeeded"}
    ]


def test_history_retains_requests_and_outcomes_with_a_bounded_context(http, client, seeded):
    setup, assets = seeded
    planner = http.app.state.services.assistants.provider
    body = dict(project_id=setup["project_id"], context={"asset_id": assets[0]["id"]})
    conversation = None
    for index in range(12):
        planner.queue = [tool("select_annotation_model", model_id=setup["baseline_id"])]
        reply = client.post(
            "/api/assistant",
            dict(
                body,
                message=f"Use baseline {index}: " + "annotation context " * 100,
                conversation_id=conversation,
            ),
        )
        conversation = reply["conversation_id"]
    messages = planner.calls[-1][0]
    assert sum(len(m.model_dump_json()) for m in messages[1:-1]) < 8000
    assert not any(m.tool_calls or m.role == "tool" for m in messages)
    assert [m.role for m in messages[1:-1]] == ["user", "assistant"] * ((len(messages) - 2) // 2)
    assert all("Selected" in m.content for m in messages[1:-1] if m.role == "assistant")
    from monailabel.core.chat import Conversation

    stored = http.app.state.services.store.get(Conversation, conversation)
    assert all(any(m.tool_calls for m in turn.messages) for turn in stored.turns)
    assert messages[-1].content.startswith("Use baseline 11:")
    assert not any(m.content.startswith("Use baseline 0:") for m in messages)


def test_local_variants_reserve_context_for_their_reasoning_budget():
    assert CoordinatorConfig(variant="lightning").max_tokens == 8192
    assert CoordinatorConfig(variant="4b").max_tokens == 4096
    assert CoordinatorConfig(variant="9b").max_tokens == 4096
    assert CoordinatorConfig(variant="lightning", max_tokens=2048).max_tokens == 2048


def test_responses_timeout_allows_reasoning_without_overriding_explicit_limits():
    assert CoordinatorConfig(provider="openai", model="gpt-6-astra").timeout == 180
    assert CoordinatorConfig(provider="openai", model="azure/openai/gpt-6-astra").timeout == 180
    assert CoordinatorConfig(provider="openai", model="gpt-6-astra", timeout=45).timeout == 45
    assert CoordinatorConfig(provider="local").timeout == 90
