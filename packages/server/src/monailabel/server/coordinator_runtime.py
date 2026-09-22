"""Managed local model serving; reuse cached weights and healthy local runtimes."""

import hashlib
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
from filelock import FileLock

from monailabel.core.chat import ChatMessage, ToolDefinition
from monailabel.core.errors import DomainError
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.http import HttpChat
from monailabel.providers.chat.local_models import LOCAL_MODELS

log = logging.getLogger(__name__)


class CoordinatorRuntime:
    def __init__(self, config: CoordinatorConfig):
        if (
            config.provider == "local"
            and "timeout" not in config.model_fields_set
            and platform.system() == "Linux"
            and platform.machine() in {"aarch64", "arm64"}
        ):
            # Nano's 4,096-token reasoning budget can exceed 90 seconds on
            # GB10. Include prefill time without overriding an explicit limit.
            config = config.model_copy(update={"timeout": 240})
        self.config = config
        self.profile = LOCAL_MODELS[config.variant]
        self.cache = (
            Path(os.environ.get("MONAILABEL_CACHE_DIR", str(Path.home() / ".cache/monailabel")))
            / "coordinator"
        )
        self.state = "configured" if config.provider != "local" else "starting"
        self.detail = (
            "Hosted coordinator configured."
            if config.provider != "local"
            else "Preparing local Nemotron."
        )
        suffix = hashlib.sha256(str(self.cache.resolve()).encode()).hexdigest()[:10]
        self.name = f"monailabel-coordinator-{config.variant}-{suffix}"
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.http = HttpChat(config, key=self.local_key if config.provider == "local" else None)

    def start(self) -> None:
        if self.config.provider == "local":
            self.thread = threading.Thread(
                target=self.prepare, name="coordinator-setup", daemon=True
            )
            self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=3)

    def status(self) -> dict[str, str]:
        return {
            "provider": self.config.provider,
            "model": self.profile.repository
            if self.config.provider == "local"
            else self.config.model_name,
            "state": self.state,
            "message": self.detail,
        }

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        *,
        require_tool: bool = False,
    ) -> ChatMessage:
        if self.config.provider == "local" and self.state != "ready":
            raise DomainError(self.detail, code="coordinator_unavailable", status=503)
        return self.http.complete(messages, tools, require_tool=require_tool)

    def local_key(self) -> str:
        return (self.cache / "runtime.key").read_text().strip()

    @staticmethod
    def _run(args: list[str], *, timeout: int = 60, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode:
            raise RuntimeError(
                "Docker command failed. Check Docker/NVIDIA GPU support and "
                "the coordinator setup log."
            )
        return result.stdout.strip()

    def _ready(self) -> bool:
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.get(
                    self.config.endpoint + "/models",
                    headers={"Authorization": "Bearer " + self.local_key()},
                )
                return response.status_code == 200 and any(
                    item["id"] == self.profile.name for item in response.json()["data"]
                )
        except (httpx.HTTPError, OSError, ValueError, KeyError):
            return False

    def _reuse_lightning(self) -> bool:
        """Recognize the workstation's shared SGLang service without managing its lifecycle."""
        if (
            self.config.variant != "lightning"
            or self.config.gpu != "0"
            or self.config.gpu_memory_utilization is not None
        ):
            return False
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                info = client.get("http://127.0.0.1:8001/get_model_info")
                if (
                    info.status_code != 200
                    or info.json().get("model_path") != self.profile.repository
                ):
                    return False
                models = client.get("http://127.0.0.1:8001/v1/models").json()["data"]
                if len(models) != 1:
                    return False
                config = CoordinatorConfig(
                    provider="compatible",
                    model=models[0]["id"],
                    base_url="http://127.0.0.1:8001/v1",
                    thinking=self.config.thinking is not False,
                    temperature=0,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                self.http = HttpChat(config)
                self.detail = (
                    "Reusing the shared local Lightning service on port 8001; "
                    "its memory settings are unchanged."
                )
                return True
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def prepare(self) -> None:
        """Prepare synchronously for CLI checks; server startup uses a background thread."""
        try:
            if self._reuse_lightning():
                self.state = "ready"
                return
            if not shutil.which("docker"):
                raise RuntimeError(
                    "Local Nemotron needs Docker with NVIDIA GPU support. Install "
                    "it or use --assistant compatible with an existing endpoint."
                )
            self.cache.mkdir(parents=True, exist_ok=True)
            self.cache.chmod(0o700)
            with FileLock(str(self.cache / "setup.lock"), timeout=1800):
                key_path = self.cache / "runtime.key"
                if not key_path.exists():
                    with key_path.open("x") as stream:
                        key_path.chmod(0o600)
                        stream.write(secrets.token_urlsafe(48))
                self._provision()
                self.state, self.detail = "ready", f"Local Nemotron {self.config.variant} is ready."
                log.info(self.detail)
        except Exception as error:
            self.state = "error"
            self.detail = (
                str(error)
                if isinstance(error, RuntimeError)
                else "Local coordinator setup failed. See " + str(self.cache / "setup.log")
            )
            log.error(self.detail)

    def _hub(self) -> Path:
        global_hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        repository = "models--" + self.profile.repository.replace("/", "--")
        model_cache = global_hub / "hub" / repository
        snapshot = model_cache / "snapshots" / self.profile.revision
        # A shared/root-owned cache can contain a partial snapshot. Downloads still
        # need to create locks and metadata; never change that cache's permissions.
        writable = all(
            not path.exists() or os.access(path, os.W_OK)
            for path in (
                global_hub,
                global_hub / "hub",
                model_cache,
                model_cache / "blobs",
                snapshot,
                global_hub / "hub/.locks",
                global_hub / "hub/.locks" / repository,
            )
        )
        return global_hub if snapshot.is_dir() and writable else self.cache / "huggingface"

    @property
    def memory_fraction(self) -> float:
        if self.config.gpu_memory_utilization is not None:
            return self.config.gpu_memory_utilization
        if platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}:
            # Spark shares RAM with viewers and annotation/training models. This
            # is a serving budget, not a reservation for every GPU workload.
            return {"4b": 0.15, "9b": 0.25, "lightning": 0.30}[self.config.variant]
        return 0.8

    def _download(self, hub: Path, user_args: list[str]) -> None:
        self.detail = (
            f"Preparing Nemotron {self.config.variant}. First download/startup "
            f"can take several minutes."
        )
        log.info(self.detail)
        hub.mkdir(parents=True, exist_ok=True)
        download = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download({self.profile.repository!r}, revision={self.profile.revision!r}, "
            "ignore_patterns=['modeling_nemotron_h.py'])"
        )
        args = [
            "docker",
            "run",
            "--rm",
            *user_args,
            "--entrypoint",
            "python",
            "--env",
            "HF_HOME=/cache",
            "--volume",
            f"{hub.resolve()}:/cache",
            "--env",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
            self.profile.image,
            "-c",
            download,
        ]
        with (self.cache / "setup.log").open("a") as stream:
            process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if self.stopping.wait(1):
                    process.terminate()
                    process.wait(timeout=10)
                    raise RuntimeError(
                        "Coordinator setup stopped; restart to resume cached downloads."
                    )
            if process.returncode:
                raise RuntimeError(
                    "Coordinator download failed. See " + str(self.cache / "setup.log")
                )

    def _engine(self, snapshot: str, hub: Path) -> tuple[str, list[str]]:
        if self.profile.engine == "sglang":
            # Resolve the credential inside the container, so it never appears in host argv.
            # The pinned server logs its arguments dataclass, including api_key.
            # Redact its repr before startup without changing authentication.
            launcher = (
                "import os,sys,runpy; "
                "from sglang.srt.server_args import ServerArgs; "
                "_original_repr=ServerArgs.__repr__; "
                "ServerArgs.__repr__=lambda self:_original_repr(self).replace("
                "repr(os.environ['VLLM_API_KEY']),repr('<redacted>')); "
                "sys.argv=['sglang.launch_server',*sys.argv[1:],'--api-key',"
                "os.environ['VLLM_API_KEY']]; "
                "runpy.run_module('sglang.launch_server',run_name='__main__')"
            )
            return "python", [
                "-c",
                launcher,
                "--model-path",
                snapshot,
                "--served-model-name",
                self.profile.name,
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--trust-remote-code",
                "--context-length",
                "32768",
                "--max-total-tokens",
                "32768",
                "--mem-fraction-static",
                str(self.memory_fraction),
                "--max-running-requests",
                "4",
                "--cuda-graph-max-bs-decode",
                "4",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "nemotron_3",
            ]
        args = [
            "serve",
            snapshot,
            "--served-model-name",
            self.profile.name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-num-seqs",
            "4",
            "--max-model-len",
            "16384",
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(self.memory_fraction),
            "--kv-cache-memory-bytes",
            "1073741824",
            "--trust-remote-code",
            "--mamba_ssm_cache_dtype",
            "float32",
            "--enable-auto-tool-choice",
        ]
        if self.config.variant == "4b":
            args += [
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser-plugin",
                snapshot + "/nano_v3_reasoning_parser.py",
                "--reasoning-parser",
                "nano_v3",
            ]
        else:
            shutil.copyfile(
                Path(__file__).parent / "resources/nemotron9_parser.py.txt",
                hub / "nemotron9_parser.py",
            )
            args += [
                "--tool-parser-plugin",
                "/cache/nemotron9_parser.py",
                "--tool-call-parser",
                "nemotron_json",
            ]
        return "vllm", args

    def _provision(self) -> None:
        config_id = hashlib.sha256(
            json.dumps(
                [
                    self.profile.image,
                    self.profile.revision,
                    self.config.gpu,
                    self.memory_fraction,
                    "bounded-serving-v3",
                ]
            ).encode()
        ).hexdigest()
        inspection = subprocess.run(
            ["docker", "inspect", self.name], capture_output=True, text=True, timeout=30
        )
        if inspection.returncode == 0:
            info = json.loads(inspection.stdout)[0]
            if info["Config"].get("Labels", {}).get("monailabel.coordinator") != config_id:
                raise RuntimeError(
                    f"Cached coordinator configuration differs. Stop/remove {self.name} "
                    f"before changing GPU/runtime."
                )
            if not info["State"]["Running"]:
                self._run(["docker", "start", self.name])
        else:
            hub = self._hub()
            user_args = ["--user", f"{os.getuid()}:{os.getgid()}"] if os.name != "nt" else []
            self._download(hub, user_args)
            repository_path = self.profile.repository.replace("/", "--")
            snapshot = f"/cache/hub/models--{repository_path}/snapshots/{self.profile.revision}"
            executable, args = self._engine(snapshot, hub)
            environment = dict(os.environ, VLLM_API_KEY=self.local_key())
            self._run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    self.name,
                    "--label",
                    f"monailabel.coordinator={config_id}",
                    "--gpus",
                    f"device={self.config.gpu}",
                    "--shm-size",
                    "2g",
                    "--publish",
                    f"127.0.0.1:{self.profile.port}:8000",
                    *user_args,
                    "--volume",
                    f"{hub.resolve()}:/cache",
                    "--env",
                    "VLLM_API_KEY",
                    "--env",
                    "HOME=/cache",
                    "--env",
                    "HF_HOME=/cache",
                    "--env",
                    "HF_HUB_OFFLINE=1",
                    "--env",
                    "VLLM_CACHE_ROOT=/cache/vllm",
                    "--env",
                    "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm",
                    "--entrypoint",
                    executable,
                    self.profile.image,
                    *args,
                ],
                timeout=120,
                env=environment,
            )
        self.detail = "Loading Nemotron on the GPU."
        # Fresh ARM64/GB10 kernels can take several minutes to compile after
        # weights are loaded; keep readiness bounded without abandoning JIT.
        deadline = time.monotonic() + 1800
        while not self._ready():
            if self.stopping.wait(2):
                raise RuntimeError(
                    "Coordinator startup stopped. The cached runtime can be reused next time."
                )
            runtime_info = json.loads(self._run(["docker", "inspect", self.name]))
            if not runtime_info[0]["State"]["Running"] or time.monotonic() > deadline:
                raise RuntimeError(
                    f"Coordinator did not become ready. Inspect with: docker logs {self.name}"
                )
