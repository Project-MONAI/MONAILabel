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

"""Prepared inference remains usable; cache misses never provision visitor resources."""

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord
from monailabel.core.preparation import prepared_models, prepared_models_only


@pytest.fixture(params=["sam2", "medsam2", "vista3d", "nninteractive", "totalsegmentator-ct"])
def cache(request, tmp_path, monkeypatch):
    root = tmp_path / "models"
    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(root))
    data = b"synthetic pinned weight bytes"
    checksum = hashlib.sha256(data).hexdigest()
    name = request.param
    if name in {"sam2", "medsam2"}:
        from monailabel.sam import weights

        monkeypatch.setitem(
            weights.MODELS, name, replace(weights.MODELS[name], size=len(data), checksum=checksum)
        )
        path = weights.checkpoint_path(name)
        call, ready = lambda: weights.weights(name), lambda: weights.prepared(name)
    elif name == "vista3d":
        from monailabel.monai import vista_weights as weights

        monkeypatch.setattr(weights, "WEIGHT_BYTES", len(data))
        monkeypatch.setattr(weights, "CHECKSUM", checksum)
        path = weights.checkpoint_path()
        call, ready = weights.pretrained_weights, weights.prepared
    elif name == "nninteractive":
        from monailabel.nninteractive import weights

        monkeypatch.setattr(weights, "FILES", {"fold_0/model.pth": (len(data), checksum)})
        path = weights.cache_root() / weights.REVISION / "fold_0/model.pth"
        call, ready = lambda: weights.weights(lambda _: None), weights.prepared
    else:
        from monailabel.totalsegmentator import weights

        monkeypatch.setattr(
            weights,
            "manifest",
            lambda _: {
                "sha256": checksum,
                "folder": "model",
                "files": {"model/checkpoint.pth": checksum},
            },
        )
        path = weights.cache_root(name) / "model/checkpoint.pth"
        call, ready = lambda: weights.pretrained_weights(name), lambda: weights.prepared(name)
    network = Mock(side_effect=AssertionError("A visitor cannot download weights"))
    monkeypatch.setattr(httpx, "stream", network)
    return SimpleNamespace(root=root, path=path, call=call, ready=ready, data=data, network=network)


def test_missing_weights_do_not_create_cache_locks_or_downloads(cache):
    with prepared_models(), pytest.raises(DomainError) as error:
        cache.call()
    assert error.value.status == 503 and error.value.code == "not_prepared"
    assert not cache.root.exists()
    cache.network.assert_not_called()


def test_prepared_weights_are_read_only_and_checksum_verified_on_every_use(cache):
    cache.path.parent.mkdir(parents=True)
    cache.path.write_bytes(cache.data)
    original = set(cache.root.rglob("*"))
    with prepared_models():
        assert cache.ready()
        assert cache.call().exists()
        assert set(cache.root.rglob("*")) == original
        # Same size, different bytes. A previous successful lookup must not bypass verification.
        cache.path.write_bytes(b"x" * len(cache.data))
        with pytest.raises(DomainError):
            cache.call()
    assert set(cache.root.rglob("*")) == original
    cache.network.assert_not_called()


def test_file_removed_after_catalog_preflight_does_not_trigger_repair(cache):
    cache.path.parent.mkdir(parents=True)
    cache.path.write_bytes(cache.data)
    assert cache.ready()
    cache.path.unlink()
    before = set(cache.root.rglob("*"))
    with prepared_models(), pytest.raises(DomainError):
        cache.call()
    assert set(cache.root.rglob("*")) == before
    assert not cache.path.exists()
    cache.network.assert_not_called()


def test_catalog_checks_cache_presence_without_hashing_or_loading_frameworks(
    cache, monkeypatch, tmp_path
):
    from monailabel.server.models.preparation import prepared
    from monailabel.server.storage import Artifacts

    # Framework availability is a separate host preparation input. No framework
    # or model is loaded merely to enumerate the catalog.
    monkeypatch.setattr("monailabel.server.models.preparation.find_spec", lambda _: object())
    monkeypatch.setenv("MONAILABEL_NNINTERACTIVE_PYTHON", sys.executable)
    provider = cache.path.relative_to(cache.root).parts[0]
    if provider == "totalsegmentator":
        provider = "totalsegmentator-ct"
    model = ModelRecord(
        project_id="test", name="Prepared fixture", provider=provider, label_ids=[0], read_only=True
    )
    artifacts = Artifacts(tmp_path / "disposable-artifacts")
    assert not prepared(model, artifacts)
    cache.path.parent.mkdir(parents=True)
    cache.path.write_bytes(cache.data)
    assert prepared(model, artifacts)
    if provider == "nninteractive":
        monkeypatch.delenv("MONAILABEL_NNINTERACTIVE_PYTHON")
        assert not prepared(model, artifacts)
    else:
        monkeypatch.setattr("monailabel.server.models.preparation.find_spec", lambda _: None)
        assert not prepared(model, artifacts)
    cache.network.assert_not_called()


@pytest.mark.parametrize("provider", ["vista3d", "monai-unet", "nnunet-v2"])
def test_derived_models_use_prepared_project_artifacts_not_unneeded_base_downloads(
    provider, tmp_path, monkeypatch
):
    from monailabel.server.models.preparation import prepared
    from monailabel.server.storage import Artifacts

    monkeypatch.setenv("MONAILABEL_MODELS_DIR", str(tmp_path / "missing-base-cache"))
    monkeypatch.setattr("monailabel.server.models.preparation.find_spec", lambda _: object())
    artifacts = Artifacts(tmp_path / "artifacts")
    key = artifacts.put(b"synthetic derived checkpoint")
    state = {"checkpoint_key": key}
    if provider == "nnunet-v2":
        state |= dict.fromkeys(("plans_key", "dataset_key", "fingerprint_key"), key)
    model = ModelRecord(
        project_id="test",
        name="Derived",
        provider=provider,
        label_ids=[0],
        state_key=artifacts.put_json(state),
    )
    assert prepared(model, artifacts)
    artifacts.path(key).unlink()
    assert not prepared(model, artifacts)
    assert not (tmp_path / "missing-base-cache").exists()


def test_context_cannot_be_relaxed_or_leak_between_threads(monkeypatch):
    monkeypatch.delenv("MONAILABEL_PREPARED_MODELS_ONLY", raising=False)
    assert not prepared_models_only()
    with pytest.raises(RuntimeError), prepared_models():
        with prepared_models(False):
            assert prepared_models_only()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(prepared_models_only).result(timeout=5) is False
        raise RuntimeError("Failed operation")
    assert not prepared_models_only()
    monkeypatch.setenv("MONAILABEL_PREPARED_MODELS_ONLY", "1")
    with prepared_models(False):
        assert prepared_models_only()


def test_nninteractive_missing_runtime_never_invokes_uv(tmp_path, monkeypatch):
    from monailabel.nninteractive import runtime

    monkeypatch.delenv("MONAILABEL_NNINTERACTIVE_PYTHON", raising=False)
    process = Mock(side_effect=AssertionError("No dependency installer is allowed"))
    monkeypatch.setattr(runtime.subprocess, "Popen", process)
    with prepared_models(), pytest.raises(DomainError, match="runtime is not prepared"):
        runtime.run_worker(tmp_path, lambda _: None)
    assert not list(tmp_path.iterdir())
    process.assert_not_called()


def test_nninteractive_prepared_runtime_executes_directly_without_installer(tmp_path, monkeypatch):
    from monailabel.nninteractive import runtime

    monkeypatch.setenv("MONAILABEL_NNINTERACTIVE_PYTHON", sys.executable)
    popen = Mock(return_value=SimpleNamespace(returncode=0, wait=lambda **_: 0, poll=lambda: 0))
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    with prepared_models():
        runtime.run_worker(tmp_path, lambda _: None)
    command = popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-I", "-c"]
    assert "uv" not in command and "--script" not in command
    assert command[-1] == str(tmp_path)
    assert "PYTHONPATH" not in popen.call_args.kwargs["env"]


def test_nninteractive_wrong_prepared_versions_fail_without_running_worker(tmp_path, monkeypatch):
    from monailabel.nninteractive.environment import worker_command

    monkeypatch.setenv("MONAILABEL_NNINTERACTIVE_PYTHON", sys.executable)
    # Inject only metadata lookup in the test bootstrap; the real interpreter still
    # executes the production version gate, which must exit before runpy.
    command = worker_command()
    command[-1] = command[-1].replace(
        "from importlib.metadata import version;", "version=lambda name:'0.0.0';"
    )
    result = subprocess.run(
        command + [str(tmp_path / "must-not-run.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
        env=os.environ | {"UV_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "versions differ" in result.stderr and "must-not-run" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_nninteractive_matching_versions_run_the_prepared_script(tmp_path, monkeypatch):
    from monailabel.nninteractive.environment import worker_command

    monkeypatch.setenv("MONAILABEL_NNINTERACTIVE_PYTHON", sys.executable)
    command = worker_command()
    command[-1] = command[-1].replace(
        "from importlib.metadata import version;",
        "version=lambda name:{'nninteractive':'2.6.0','nnunetv2':'2.8.1',"
        "'torch':'2.11.0+cu130'}[name];",
    )
    script = tmp_path / "synthetic-worker.py"
    script.write_text("import sys; assert sys.flags.isolated; print(sys.argv[1])")
    result = subprocess.run(
        command + [str(script), str(tmp_path)], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path)
