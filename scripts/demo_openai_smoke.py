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

"""Bounded GPT assistant tool round-trip using disposable example metadata only.

Offline by default. --execute permits at most two paid Responses requests, with
no retries, to the fixed OpenAI endpoint. A private output directory contains
an exclusive marker so repeating the same command cannot repeat paid requests.
No running workspace is opened and no application service is restarted.
"""

import argparse
import json
import logging
import os
import resource
import stat
import time
from pathlib import Path

import httpx
from demo_openai_run import read_key

from monailabel.core.chat import ChatMessage, ToolDefinition
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.http import HttpChat

ENDPOINT = "https://api.openai.com/v1/responses"
MODELS = ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="gpt-6-luna")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = args.output.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit("Use an owner-only private output directory.")
    key = read_key(args.key_file) if args.execute and args.key_file else "offline-only"
    if args.execute:
        if args.key_file is None:
            parser.error("--execute requires --key-file")
        marker = os.open(args.output / "paid-started", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(marker)
    report = {
        "model": args.model,
        "live": args.execute,
        "requests": 0,
        "usage": [],
        "max_output_tokens_per_request": 1024,
        "automatic_retries": 0,
        "live_workspace_accessed": False,
        "passed": False,
    }

    class GuardedTransport(httpx.BaseTransport):
        def __init__(self):
            self.transport = httpx.HTTPTransport(retries=0) if args.execute else None

        def handle_request(self, request):
            body = json.loads(request.content)
            if (
                str(request.url) != ENDPOINT
                or request.method != "POST"
                or report["requests"] >= 2
                or body.get("model") != args.model
                or body.get("max_output_tokens") != 1024
                or body.get("store") is not False
                or len(request.content) > 16000
            ):
                raise RuntimeError("Request outside the approved small test.")
            report["requests"] += 1
            if self.transport:
                response = self.transport.handle_request(request)
                response.read()
            elif report["requests"] == 1:
                response = httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "offline-1",
                                "name": "list_demo_examples",
                                "arguments": "{}",
                            }
                        ],
                    },
                )
            else:
                response = httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            "Use QuPath for pathology, CVAT for video "
                                            "and Slicer or OHIF for radiology."
                                        ),
                                    }
                                ],
                            }
                        ],
                    },
                )
            if response.status_code == 200:
                usage = response.json().get("usage", {})
                report["usage"].append(
                    {
                        k: usage[k]
                        for k in ("input_tokens", "output_tokens", "total_tokens")
                        if isinstance(usage.get(k), int)
                    }
                )
            return response

        def close(self):
            # HttpChat creates a client per turn. Keep this bounded test transport
            # alive until both turns finish, then explicitly close it below.
            pass

    transport = GuardedTransport()
    provider = HttpChat(
        CoordinatorConfig(
            provider="openai", model=args.model, max_tokens=1024, thinking=False, timeout=60
        ),
        key=lambda: key,
        transport=transport,
    )
    tool = ToolDefinition(
        name="list_demo_examples",
        description="Read the prepared demo example list.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You help users operate MONAI Label. Call the read-only tool, "
                "then explain its viewer choices briefly. Never claim to have run segmentation."
            ),
        ),
        ChatMessage(
            role="user", content="Which demo examples can I try, and which viewers should I use?"
        ),
    ]
    started = time.monotonic()
    try:
        first = provider.complete(messages, [tool], require_tool=True)
        assert len(first.tool_calls) == 1
        call = first.tool_calls[0]
        assert call.name == tool.name and call.arguments == {}
        messages.extend(
            [
                first,
                ChatMessage(
                    role="tool",
                    tool_call_id=call.id,
                    content=json.dumps(
                        {
                            "radiology": {"viewers": ["Slicer", "OHIF"], "model": "nnInteractive"},
                            "pathology": {"viewer": "QuPath", "model": "SAM 2.1"},
                            "endoscopy": {"viewer": "CVAT", "model": "SAM 2.1"},
                        }
                    ),
                ),
            ]
        )
        reply = provider.complete(messages, [])
        assert (
            not reply.tool_calls
            and "qupath" in reply.content.lower()
            and "cvat" in reply.content.lower()
        )
        report.update(
            passed=True,
            typed_tool_call=True,
            tool_result_understood=True,
            seconds=round(time.monotonic() - started, 2),
        )
    except Exception as error:
        # No traceback, upstream body, request headers, or credential is persisted.
        report["error_type"] = type(error).__name__
    finally:
        if transport.transport:
            transport.transport.close()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
