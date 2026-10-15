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

"""Offline coordinator imports retain image pins and model content verification."""

import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_prepare_coordinator.py"


@pytest.mark.parametrize("tamper", [None, "layers", "config", "index", "manifest"])
def test_archive_import_checks_pinned_content(tmp_path, monkeypatch, tamper):
    module = runpy.run_path(str(SCRIPT))
    config = {
        "config": {"Env": ["PUBLIC_FIXTURE=1"]},
        "rootfs": {"diff_ids": ["sha256:original-layer"]},
        "os": "linux",
        "architecture": "amd64",
    }
    content = json.dumps(config).encode()
    config_hash = hashlib.sha256(content).hexdigest()
    archive = tmp_path / "image.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo(config_hash + ".json")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config": {"digest": "sha256:" + config_hash}}))
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "manifests": [
                    {
                        "platform": {"os": "linux", "architecture": "amd64"},
                        "digest": "sha256:" + module["digest"](manifest),
                    }
                ]
            }
        )
    )
    monkeypatch.setitem(
        module["LOCAL_MODELS"],
        "lightning",
        SimpleNamespace(image="fixture@sha256:" + module["digest"](index)),
    )
    loaded = {
        "Id": "sha256:docker-imported-manifest",
        "Config": config["config"],
        "RootFS": {"Layers": config["rootfs"]["diff_ids"]},
        "Os": "linux",
        "Architecture": "amd64",
    }
    if tamper == "layers":
        loaded["RootFS"] = {"Layers": ["sha256:different-layer"]}
    elif tamper == "config":
        loaded["Config"] = {"Env": ["CHANGED=1"]}
    elif tamper in {"index", "manifest"}:
        (index if tamper == "index" else manifest).write_text("{}")

    def inspect(command, **kwargs):
        assert command == ["docker", "image", "inspect", "fixture:imported"]
        return SimpleNamespace(stdout=json.dumps([loaded]))

    monkeypatch.setattr(module["subprocess"], "run", inspect)
    if tamper:
        with pytest.raises(RuntimeError):
            module["verified_image"](index, manifest, "fixture:imported", archive)
    else:
        assert (
            module["verified_image"](index, manifest, "fixture:imported", archive) == loaded["Id"]
        )


@pytest.mark.parametrize("global_cache", [False, True])
def test_snapshot_verifies_original_git_and_lfs_hashes(tmp_path, monkeypatch, global_cache):
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.setitem(
        module["LOCAL_MODELS"],
        "lightning",
        SimpleNamespace(repository="fixture/model", revision="pinned"),
    )
    root = tmp_path / "hub/models--fixture--model"
    snapshot = root / "snapshots/pinned"
    snapshot.mkdir(parents=True)
    (root / "blobs").mkdir()
    contents = {
        "config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors.index.json": b'{"weight_map":{"weight":"weights.safetensors"}}',
        "weights.safetensors": b"synthetic weights",
    }
    for name, content in contents.items():
        checksum = (
            hashlib.sha256(content).hexdigest()
            if name.endswith(".safetensors")
            else hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        )
        blob = root / "blobs" / checksum
        if not blob.exists():
            blob.write_bytes(content)
        if global_cache and name.endswith(".safetensors"):
            global_blob = tmp_path / "hub/blobs/fixture/global-content-id"
            global_blob.parent.mkdir(parents=True)
            blob.rename(global_blob)
            blob.symlink_to(global_blob)
        (snapshot / name).symlink_to(blob)
    module["verify_snapshot"](tmp_path)
    (snapshot / "weights.safetensors").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        module["verify_snapshot"](tmp_path)
