"""Vendor-independent conversation and tool-calling contracts."""

from typing import Literal, Protocol

from pydantic import Field, JsonValue

from monailabel.core.models import AssistantReply, Contract, Record


class ToolDefinition(Contract):
    name: str
    description: str
    parameters: dict[str, JsonValue]


class ToolCall(Contract):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, JsonValue]


class ChatMessage(Contract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    tool_call_id: str | None = None
    # Opaque provider continuity data, e.g. Gemini thought signatures. Never executed.
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ChatProvider(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        require_tool: bool = False,
    ) -> ChatMessage: ...


class ChatTurn(Contract):
    request_id: str
    request_hash: str
    messages: list[ChatMessage]
    reply: AssistantReply


class Conversation(Record):
    user_id: str
    project_id: str | None = None
    asset_id: str | None = None
    turns: list[ChatTurn] = Field(default_factory=list)


class ToolExecution(Record):
    """Durable receipt prevents repeating a tool after an interrupted HTTP request."""

    conversation_id: str
    project_id: str | None = None
    asset_id: str | None = None
    request_hash: str
    reply: AssistantReply | None = None
