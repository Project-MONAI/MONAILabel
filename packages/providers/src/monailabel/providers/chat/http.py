"""Tool calling through OpenAI-compatible Chat Completions or native Claude Messages."""

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
        payload = self._claude(messages, tools) if native else self._openai(messages, tools)
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
        url = self.config.endpoint + ("/messages" if native else "/chat/completions")
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
            if reason in {"length", "max_tokens"}:
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
            parsed = self._parse_claude(result) if native else self._parse_openai(result)
            if require_tool and not parsed.tool_calls:
                raise ValueError("The coordinator omitted its required tool call.")
            return parsed
        except httpx.HTTPStatusError as error:
            raise DomainError(
                f"Coordinator endpoint returned HTTP {error.response.status_code}. "
                "Check its model, API base URL, credentials and tool-calling support.",
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
