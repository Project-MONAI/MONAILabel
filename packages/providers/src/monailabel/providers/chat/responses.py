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

"""OpenAI Responses serialization with stateless tool-call continuity."""

import json
from typing import Any

from monailabel.core.chat import ChatMessage, ToolCall, ToolDefinition

from .config import CoordinatorConfig


def response_payload(
    config: CoordinatorConfig, messages: list[ChatMessage], tools: list[ToolDefinition]
) -> dict[str, Any]:
    wire: list[Any] = []
    for message in messages:
        if message.role == "tool":
            wire.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        elif message.role == "assistant" and message.metadata.get("openai_response_output"):
            # Preserve encrypted reasoning, function calls and assistant phase together.
            wire.extend(message.metadata["openai_response_output"])  # type: ignore[arg-type]
        else:
            if message.content:
                wire.append({"role": message.role, "content": message.content})
            wire.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
                for call in message.tool_calls
            )
    result: dict[str, Any] = {
        "model": config.model_name,
        "input": wire,
        "store": False,
        "max_output_tokens": config.max_tokens,
        "parallel_tool_calls": False,
        # Domain tools retain optional arguments and validate them before execution.
        "tools": [{"type": "function", "strict": False, **tool.model_dump()} for tool in tools],
    }
    if config.model_name.rsplit("/", 1)[-1].startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        result["reasoning"] = {"effort": "medium" if config.thinking else "low"}
    elif config.temperature is not None:
        result["temperature"] = config.temperature
    return result


def parse_response(result: dict[str, Any]) -> ChatMessage:
    if result.get("status") != "completed":
        raise ValueError("Unfinished response")
    output = result["output"]
    content = []
    calls = []
    for item in output:
        if item["type"] == "function_call":
            if item.get("status", "completed") != "completed":
                raise ValueError("Unfinished tool call")
            calls.append(
                ToolCall(
                    id=item["call_id"],
                    name=item["name"],
                    arguments=json.loads(item["arguments"]),
                )
            )
        elif item["type"] == "message":
            if item.get("status", "completed") != "completed":
                raise ValueError("Unfinished message")
            for block in item["content"]:
                if block["type"] == "output_text":
                    content.append(block["text"])
                elif block["type"] == "refusal":
                    content.append(block["refusal"])
                else:
                    raise ValueError("Unsupported response content")
        elif item["type"] != "reasoning":
            raise ValueError("Unsupported response item")
    if not content and not calls:
        raise ValueError("Empty response")
    return ChatMessage(
        role="assistant",
        content="\n".join(content),
        tool_calls=calls,
        metadata={"openai_response_output": output},
    )
