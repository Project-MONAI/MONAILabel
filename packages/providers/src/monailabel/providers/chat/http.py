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

"""Tool calling through native OpenAI/Claude APIs or compatible Chat Completions."""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from monailabel.core.chat import ChatMessage, ToolCall, ToolDefinition
from monailabel.core.errors import DomainError

from .config import CoordinatorConfig
from .responses import parse_response, response_payload

logger = logging.getLogger(__name__)


class HttpChat:
    def __init__(
        self,
        config: CoordinatorConfig,
        *,
        key: Callable[[], str | None] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = config
        self.key = key or self._environment_key
        self.transport = transport

    def _environment_key(self) -> str | None:
        name = self.config.credential_env
        value = os.environ.get(name) if name else None
        if name and not value:
            raise DomainError(
                f"Coordinator credential is missing. Set {name} and restart MONAI Label.",
                code="coordinator_unavailable",
                status=503,
            )
        return value

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        require_tool: bool = False,
    ) -> ChatMessage:
        key = self.key()
        native = self.config.provider == "anthropic"
        responses = self.config.provider == "openai"
        if native:
            payload = self._claude(messages, tools)
        elif responses:
            payload = response_payload(self.config, messages, tools)
        else:
            payload = self._openai(messages, tools)
        if require_tool:
            if not tools:
                raise DomainError("A required tool call needs at least one available tool.")
            payload["tool_choice"] = {"type": "any"} if native else "required"
        headers = {"Content-Type": "application/json"}
        if native:
            headers["anthropic-version"] = "2023-06-01"
            if key:
                headers["x-api-key"] = key
        elif key:
            headers["Authorization"] = f"Bearer {key}"
        path = "/messages" if native else "/responses" if responses else "/chat/completions"
        url = self.config.endpoint + path
        try:
            with httpx.Client(
                timeout=self.config.timeout, transport=self.transport, follow_redirects=False
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    raise ValueError("Response is too large")
                result = response.json()
            reason = (
                result.get("stop_reason")
                if native
                else (result.get("incomplete_details") or {}).get("reason", result.get("status"))
                if responses
                else result.get("choices", [{}])[0].get("finish_reason")
            )
            usage = result.get("usage") or {}
            # Log only accounting metadata, never prompts, responses or credentials.
            token_usage = {
                name: value
                for name, value in usage.items()
                if name in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}
                and isinstance(value, int)
            }
            logger.info(
                "Coordinator model=%s finish_reason=%s tokens=%s",
                self.config.model_name,
                reason,
                token_usage,
            )
            if reason in {"length", "max_tokens", "max_output_tokens"}:
                logger.warning(
                    "Coordinator response token limit: model=%s finish_reason=%s tokens=%s",
                    self.config.model_name,
                    reason,
                    token_usage,
                )
                raise DomainError(
                    "Coordinator reached its response token limit before finishing a tool call. "
                    "No tool from this response was executed. Try a shorter request or "
                    "increase the coordinator output budget; server logs record token usage.",
                    code="coordinator_invalid_response",
                    status=502,
                )
            if native:
                parsed = self._parse_claude(result)
            elif responses:
                parsed = parse_response(result)
            else:
                parsed = self._parse_openai(result)
            if require_tool and not parsed.tool_calls:
                raise ValueError("The coordinator omitted its required tool call.")
            return parsed
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            logger.warning(
                "Coordinator request failed: provider=%s model=%s http_status=%s",
                self.config.provider,
                self.config.model_name,
                status,
            )
            guidance = (
                "The model service failed while processing the request. "
                "Check its health and runtime logs, then retry."
                if status >= 500
                else "Check its model, API base URL, credentials and tool-calling support."
            )
            if status == 429:
                # Inspect only known error codes; upstream messages may contain private data.
                try:
                    detail = error.response.json().get("error") or {}
                    exhausted = detail.get("type") == "insufficient_quota" or detail.get(
                        "code"
                    ) in {"insufficient_quota", "credit_balance_exhausted"}
                except (ValueError, AttributeError, TypeError):
                    exhausted = False
                guidance = (
                    "The API account has insufficient credit or quota. Check provider billing "
                    "and spending limits, then restart MONAI Label."
                    if exhausted
                    else "The API rate limit was reached. Wait before retrying."
                )
            raise DomainError(
                f"Coordinator endpoint returned HTTP {status}. {guidance}",
                code="coordinator_unavailable",
                status=503,
            ) from None
        except httpx.HTTPError:
            raise DomainError(
                "Cannot reach the coordinator. Check its runtime or hosted endpoint.",
                code="coordinator_unavailable",
                status=503,
            ) from None
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, ValidationError):
            raise DomainError(
                "Coordinator returned an incomplete or invalid tool-call response. "
                "No tool from this response was executed.",
                code="coordinator_invalid_response",
                status=502,
            ) from None

    def _openai(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> dict[str, Any]:
        wire = []
        for message in messages:
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = message.metadata.get("openai_tool_calls") or [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ]
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            wire.append(item)
        result: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": wire,
            "tools": [{"type": "function", "function": tool.model_dump()} for tool in tools],
            "tool_choice": "auto",
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.provider == "local":
            result.update(temperature=0, chat_template_kwargs={"enable_thinking": True})
        if self.config.temperature is not None:
            result["temperature"] = self.config.temperature
        if self.config.thinking is not None:
            result["chat_template_kwargs"] = {"enable_thinking": self.config.thinking}
        if self.config.provider == "local" and self.config.variant == "9b":
            # Nano v2 selects reasoning through its documented system directive.
            wire[0]["content"] += "\n/no_think" if self.config.thinking is False else "\n/think"
        return result

    def _parse_openai(self, result: dict[str, Any]) -> ChatMessage:
        choice = result["choices"][0]
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise ValueError("Unfinished response")
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if any(call.get("type") != "function" for call in calls):
            raise ValueError("Unsupported tool type")
        content = message.get("content") or ""
        if self.config.variant == "9b" and "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        return ChatMessage(
            role="assistant",
            content=content,
            tool_calls=[
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                )
                for call in calls
            ],
            metadata={"openai_tool_calls": calls} if calls else {},
        )

    def _claude(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> dict[str, Any]:
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "user" if message.role == "tool" else message.role
            blocks: list[dict[str, Any]] = []
            if message.role == "tool":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            elif message.metadata.get("claude_content"):
                blocks = message.metadata["claude_content"]  # type: ignore[assignment]
            else:
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    for call in message.tool_calls
                )
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": "\n\n".join(m.content for m in messages if m.role == "system"),
            "messages": wire,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
        }

    @staticmethod
    def _parse_claude(result: dict[str, Any]) -> ChatMessage:
        if result.get("stop_reason") not in {"end_turn", "tool_use"}:
            raise ValueError("Unfinished response")
        blocks = result["content"]
        return ChatMessage(
            role="assistant",
            content="\n".join(b["text"] for b in blocks if b["type"] == "text"),
            tool_calls=[
                ToolCall(id=b["id"], name=b["name"], arguments=b["input"])
                for b in blocks
                if b["type"] == "tool_use"
            ],
            metadata={"claude_content": blocks},
        )
