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

"""Managed serving budgets and cache safety without starting a GPU runtime."""

import json
import runpy
import subprocess
import sys
import types
from dataclasses import dataclass

import httpx
import pytest
from pydantic import ValidationError

from monailabel.core.chat import ChatMessage
from monailabel.core.errors import DomainError
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.coordinator_runtime import CoordinatorRuntime


@pytest.mark.parametrize("failure", [False, True])
def test_openai_readiness_requires_a_real_tool_call_without_local_services(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path / "unused"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.delenv("NV_INFERENCE_API_KEY", raising=False)
    runtime = CoordinatorRuntime(CoordinatorConfig(provider="openai", model="gpt-6-astra"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Hosted assistants must not provision or reuse local model services")

    monkeypatch.setattr(runtime, "_provision", forbidden)
    monkeypatch.setattr(runtime, "_reuse_lightning", forbidden)
    monkeypatch.setattr("shutil.which", forbidden)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        assert runtime.state == "starting"
        if failure:
            return httpx.Response(429, json={"error": {"type": "insufficient_quota"}})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "ready1",
                        "name": "readiness_probe",
                        "arguments": '{"status":"ready"}',
                    }
                ],
            },
        )

    runtime.http.transport = httpx.MockTransport(respond)
    with pytest.raises(DomainError):
        runtime.complete([ChatMessage(role="user", content="Create a project")], [])
    runtime.start()
    runtime.thread.join(timeout=5)
    assert runtime.state == ("error" if failure else "ready")
    assert len(calls) == 1
    assert calls[0]["tool_choice"] == "required"
    assert not runtime.cache.exists()
    if failure:
        assert "insufficient credit or quota" in runtime.detail
    runtime.close()


@pytest.mark.parametrize("architecture", ["aarch64", "arm64"])
@pytest.mark.parametrize("variant,fraction", [("lightning", 0.30), ("4b", 0.15), ("9b", 0.25)])
def test_shared_memory_budgets_and_bounded_concurrency(
    tmp_path, monkeypatch, architecture, variant, fraction
):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: architecture)
    runtime = CoordinatorRuntime(CoordinatorConfig(variant=variant))
    executable, args = runtime._engine("/cache/snapshot", tmp_path)
    assert runtime.memory_fraction == fraction
    if variant == "lightning":
        assert executable == "python"
        assert args[args.index("--mem-fraction-static") + 1] == str(fraction)
        assert args[args.index("--max-running-requests") + 1] == "4"
        assert args[args.index("--cuda-graph-max-bs-decode") + 1] == "4"
        assert args[args.index("--context-length") + 1] == "32768"
    else:
        assert executable == "vllm"
        assert args[args.index("--gpu-memory-utilization") + 1] == str(fraction)
        assert args[args.index("--max-num-seqs") + 1] == "4"
        assert args[args.index("--kv-cache-memory-bytes") + 1] == "1073741824"
        assert "--enforce-eager" in args


def test_discrete_gpu_default_and_explicit_override(monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert CoordinatorRuntime(CoordinatorConfig()).memory_fraction == 0.8
    monkeypatch.setenv("MONAILABEL_ASSISTANT_GPU_MEMORY_UTILIZATION", "0.4")
    assert CoordinatorRuntime(CoordinatorConfig.from_env()).memory_fraction == 0.4
    for invalid in [0, 1, -0.1, float("nan")]:
        with pytest.raises(ValidationError):
            CoordinatorConfig(gpu_memory_utilization=invalid)


def test_arm_local_timeout_covers_reasoning_without_overriding_user_choice(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="4b"))
    assert runtime.config.timeout == runtime.http.config.timeout == 240
    assert CoordinatorRuntime(CoordinatorConfig(timeout=90)).config.timeout == 90
    hosted = CoordinatorConfig(provider="openai", model="test")
    assert CoordinatorRuntime(hosted).config.timeout == 180
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert CoordinatorRuntime(CoordinatorConfig()).config.timeout == 90


def test_read_only_shared_hub_does_not_receive_downloads(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("HF_HOME", str(shared))
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path / "private"))
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="4b"))
    snapshot = (
        shared
        / "hub"
        / ("models--" + runtime.profile.repository.replace("/", "--"))
        / "snapshots"
        / runtime.profile.revision
    )
    snapshot.mkdir(parents=True)
    assert runtime._hub() == shared
    monkeypatch.setattr("os.access", lambda path, mode: path != shared)
    assert runtime._hub() == runtime.cache / "huggingface"


def test_changed_budget_does_not_reuse_or_remove_cached_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path))
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="4b", gpu_memory_utilization=0.2))
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "Config": {"Labels": {"monailabel.coordinator": "old-recipe"}},
                        "State": {"Running": True},
                    }
                ]
            ),
        )

    monkeypatch.setattr(subprocess, "run", inspect)
    with pytest.raises(RuntimeError, match="configuration differs"):
        runtime._provision()
    assert calls == [["docker", "inspect", runtime.name]]


def test_managed_runtime_supports_host_uids_absent_from_container(tmp_path, monkeypatch):
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("os.getuid", lambda: 2002)
    monkeypatch.setattr("os.getgid", lambda: 2002)
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="4b"))
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1)
    )
    monkeypatch.setattr(runtime, "_hub", lambda: tmp_path)
    monkeypatch.setattr(runtime, "_download", lambda *args: None)
    monkeypatch.setattr(runtime, "local_key", lambda: "test-runtime-key")
    monkeypatch.setattr(runtime, "_ready", lambda: True)
    calls = []
    monkeypatch.setattr(runtime, "_run", lambda command, **kwargs: calls.append(command))
    runtime._provision()
    command = calls[0]
    assert command[command.index("--user") + 1] == "2002:2002"
    environment = dict(
        command[index + 1].split("=", 1)
        for index, value in enumerate(command)
        if value == "--env" and "=" in command[index + 1]
    )
    assert environment["USER"] == environment["LOGNAME"] == "monailabel"
    for key in ("HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT"):
        assert environment[key].startswith("/cache")
    assert f"{tmp_path}:/cache" in command


@pytest.mark.parametrize("settings", [{"gpu": "1"}, {"gpu_memory_utilization": 0.4}])
def test_explicit_serving_settings_do_not_silently_reuse_a_shared_service(settings):
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="lightning", **settings))
    assert not runtime._reuse_lightning()


def readiness_response(name="readiness_probe", arguments=None):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "probe",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    {"status": "ready"} if arguments is None else arguments
                                ),
                            },
                        }
                    ]
                },
            }
        ]
    }


@pytest.mark.parametrize("failure", [None, "crash", "wrong_tool", "wrong_arguments"])
def test_shared_model_metadata_does_not_mark_failed_generation_ready(monkeypatch, failure):
    runtime = CoordinatorRuntime(CoordinatorConfig())
    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert runtime.state == "starting"
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": runtime.profile.repository})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "shared-lightning"}]})
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "shared-lightning"
        assert payload["tool_choice"] == "required"
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["readiness_probe"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if failure == "crash":
            return httpx.Response(500, text="private upstream details")
        return httpx.Response(
            200,
            json=readiness_response(
                name="create_project" if failure == "wrong_tool" else "readiness_probe",
                arguments={"status": "wrong"} if failure == "wrong_arguments" else None,
            ),
        )

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: client(
            *args, **{**kwargs, "transport": httpx.MockTransport(respond)}
        ),
    )
    monkeypatch.setattr(runtime.stopping, "wait", lambda _: True)
    runtime.prepare()
    assert calls == ["/get_model_info", "/v1/models", "/v1/chat/completions"]
    assert runtime.state == ("error" if failure else "ready")
    if failure:
        assert "tool-calling readiness check" in runtime.detail
        assert "private upstream details" not in runtime.detail


def test_managed_model_retries_generation_during_startup_without_executing_tools(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MONAILABEL_CACHE_DIR", str(tmp_path))
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="4b"))
    monkeypatch.setattr(runtime, "_provision", lambda: None)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    waits = []
    monkeypatch.setattr(runtime.stopping, "wait", lambda seconds: waits.append(seconds))
    calls = []

    def respond(request):
        assert runtime.state == "starting"
        assert runtime.detail == "Checking coordinator tool calling."
        assert request.headers["authorization"].startswith("Bearer ")
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=readiness_response())

    runtime.http.transport = httpx.MockTransport(respond)
    runtime.prepare()
    assert runtime.state == "ready"
    assert len(calls) == 2
    assert waits == [2]
    assert runtime.http.config.max_tokens == 4096
    assert runtime.http.config.thinking is None


def test_lightning_startup_repr_does_not_log_the_runtime_key(tmp_path, monkeypatch):
    key = "test-only-key-that-must-not-be-logged"
    monkeypatch.setenv("VLLM_API_KEY", key)

    @dataclass
    class ServerArgs:
        api_key: str

    module = types.ModuleType("sglang.srt.server_args")
    module.ServerArgs = ServerArgs
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", module)
    observed = []
    monkeypatch.setattr(
        runpy, "run_module", lambda *args, **kwargs: observed.append(repr(ServerArgs(key)))
    )
    monkeypatch.setattr(sys, "argv", ["launcher", "--port", "8000"])
    runtime = CoordinatorRuntime(CoordinatorConfig())
    _, args = runtime._engine("/cache/snapshot", tmp_path)
    exec(args[1], {})
    assert len(observed) == 1
    assert "api_key='<redacted>'" in observed[0]
    assert key not in observed[0]
    assert sys.argv[-2:] == ["--api-key", key]
