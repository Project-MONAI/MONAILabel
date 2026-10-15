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

"""Start the pinned local coordinator from a verified, offline Docker archive import.

For hosts with unreliable registry DNS. This does not change DNS, bypass image pins,
or expose the runtime credential. Normal MONAI Label startup can reuse the result.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.providers.chat.local_models import LOCAL_MODELS
from monailabel.server.coordinator_runtime import CoordinatorRuntime


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verified_image(index: Path, manifest: Path, image: str, archive: Path) -> str:
    profile = LOCAL_MODELS["lightning"]
    expected = profile.image.split("@sha256:")[1]
    if digest(index) != expected:
        raise RuntimeError("Image index does not match the branch's pinned image.")
    candidates = [
        entry["digest"]
        for entry in json.loads(index.read_text())["manifests"]
        if entry["platform"].get("os") == "linux"
        and entry["platform"].get("architecture") == "amd64"
    ]
    if candidates != ["sha256:" + digest(manifest)]:
        raise RuntimeError("The image manifest is not the pinned Linux AMD64 image.")
    expected_id = json.loads(manifest.read_text())["config"]["digest"]
    inspection = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(inspection.stdout)[0]
    image_id = loaded["Id"]
    # Docker's containerd image store returns an imported manifest digest as Id,
    # not the legacy config digest. Verify the original config bytes plus every
    # loaded uncompressed layer digest and the full runtime configuration.
    with tarfile.open(archive) as bundle:
        member = bundle.extractfile(expected_id.removeprefix("sha256:") + ".json")
        if member is None:
            raise RuntimeError("Archive is missing the pinned image config.")
        content = member.read()
    if "sha256:" + hashlib.sha256(content).hexdigest() != expected_id:
        raise RuntimeError("Archive image config checksum mismatch.")
    config = json.loads(content)
    if (
        loaded["Config"] != config["config"]
        or loaded["RootFS"]["Layers"] != config["rootfs"]["diff_ids"]
        or loaded["Os"] != config["os"]
        or loaded["Architecture"] != config["architecture"]
    ):
        raise RuntimeError("Loaded image filesystem or configuration differs from the pin.")
    return image_id


def verify_snapshot(hub: Path) -> None:
    profile = LOCAL_MODELS["lightning"]
    root = hub / "hub" / ("models--" + profile.repository.replace("/", "--"))
    snapshot = root / "snapshots" / profile.revision
    weight_index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    required = {"config.json", "tokenizer_config.json", *weight_index["weight_map"].values()}
    if not all((snapshot / name).is_file() for name in required):
        raise RuntimeError("The pinned model snapshot is incomplete.")
    count = 0
    for path in snapshot.rglob("*"):
        if path.is_dir():
            continue
        blob = path.resolve(strict=True)
        if not any(
            blob.is_relative_to(directory.resolve())
            for directory in (root / "blobs", hub / "hub/blobs")
        ):
            raise RuntimeError("Expected a content-addressed Hugging Face cache snapshot.")
        # Hub 2.x can deduplicate repository blobs into a global Xet blob store.
        # The snapshot link retains the original Git/LFS content checksum.
        expected = Path(os.readlink(path)).name if path.is_symlink() else path.name
        if len(expected) == 64:
            actual = digest(blob)
        elif len(expected) == 40:
            hasher = hashlib.sha1(b"blob " + str(blob.stat().st_size).encode() + b"\0")
            with blob.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            actual = hasher.hexdigest()
        else:
            raise RuntimeError("Unknown cache blob digest format.")
        if actual != expected:
            raise RuntimeError(f"Model cache checksum mismatch: {path.name}")
        count += 1
    print(f"Verified {count} cached snapshot files at revision {profile.revision}.", flush=True)


class OfflineRuntime(CoordinatorRuntime):
    def __init__(self, config: CoordinatorConfig, image_id: str):
        super().__init__(config)
        self.image_id = image_id

    def _download(self, hub: Path, user_args: list[str]) -> None:
        verify_snapshot(hub)

    def _run(self, args: list[str], *, timeout: int = 60, env: dict[str, str] | None = None) -> str:
        # Keep the original profile and lifecycle label. Only replace the registry
        # reference with its already verified exact config ID to avoid a new pull.
        args = [self.image_id if arg == self.profile.image else arg for arg in args]
        return super()._run(args, timeout=timeout, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    image_id = verified_image(args.index, args.manifest, args.image, args.archive)
    print(f"Verified pinned image: {image_id}", flush=True)
    runtime = OfflineRuntime(CoordinatorConfig(provider="local", gpu=args.gpu), image_id)
    if not args.apply:
        verify_snapshot(runtime._hub())
        print("Validation only; pass --apply to start the local GPU coordinator.")
        return
    runtime.prepare()
    print(json.dumps(runtime.status()), flush=True)
    if runtime.state != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
