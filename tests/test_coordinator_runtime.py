"""Managed serving budgets and cache safety without starting a GPU runtime."""

import json
import runpy
import subprocess
import sys
import types
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.coordinator_runtime import CoordinatorRuntime


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
    assert CoordinatorRuntime(hosted).config.timeout == 90
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


@pytest.mark.parametrize("settings", [{"gpu": "1"}, {"gpu_memory_utilization": 0.4}])
def test_explicit_serving_settings_do_not_silently_reuse_a_shared_service(settings):
    runtime = CoordinatorRuntime(CoordinatorConfig(variant="lightning", **settings))
    assert not runtime._reuse_lightning()


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
