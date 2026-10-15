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

"""Resume public demo downloads and verify the branch's pinned checksums.

Does not create accounts/projects, run inference, or contact hosted models.
Requires the installed demo environment and curl; no privileged operations.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def fetch(url: str, path: Path, algorithm: str, checksum: str) -> Path:
    def verified(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        with candidate.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest() == checksum

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not verified(path):
            raise RuntimeError(f"Existing file does not match the pinned checksum: {path}")
        print(f"Verified cache: {path}", flush=True)
        return path
    part = path.with_name(path.name + ".resume-part")
    print(f"Downloading: {path.name}", flush=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "3600",
            "--continue-at",
            "-",
            "--output",
            str(part),
            url,
        ],
        check=True,
    )
    if not verified(part):
        raise RuntimeError(f"Download checksum mismatch; retained for inspection: {part}")
    part.replace(path)
    print(f"Verified download: {path}", flush=True)
    return path


def datasets(identifiers: list[str]) -> None:
    from monailabel.server.dataset_downloads import Downloads, dataset_cache_dir, sources

    catalog = {s.id: s for s in sources() if s.importable}
    downloads = Downloads(dataset_cache_dir())
    for identifier in identifiers:
        source = catalog[identifier]
        for item in [source, *([source.mask_download] if source.mask_download else [])]:
            algorithm, checksum = item.checksum.split(":", 1)
            fetch(item.url, downloads.path(item), algorithm, checksum)
            downloads.fetch(item, SimpleNamespace(progress=lambda *args: None))


def viewer(name: str) -> None:
    from monailabel.viewers.provisioning.catalog import release
    from monailabel.viewers.provisioning.desktop import DesktopProvisioner, extract

    root = Path(os.environ["MONAILABEL_TOOLS_DIR"])
    spec = release(name)
    if spec is None or spec.build is not None:
        raise RuntimeError("This prefetch path needs a prebuilt viewer release.")
    destination = root / spec.name / f"{spec.version}-{spec.system}-{spec.machine}"
    if destination.exists():
        print(DesktopProvisioner(root).ensure(name, download=False), flush=True)
        return
    archive = fetch(
        spec.url,
        root / "downloads" / f"{spec.name}-{spec.version}.archive",
        spec.algorithm,
        spec.checksum,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".demo-install-") as temp:
        unpacked = Path(temp) / "unpacked"
        unpacked.mkdir()
        extract(archive, unpacked, spec.archive)
        candidates = list(unpacked.glob(spec.executable))
        if len(candidates) != 1 or not os.access(candidates[0], os.X_OK):
            raise RuntimeError("Viewer executable differs from the pinned recipe.")
        metadata = asdict(spec) | {"executable": candidates[0].relative_to(unpacked).as_posix()}
        (unpacked / "installation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        unpacked.rename(destination)
    print(DesktopProvisioner(root).ensure(name, download=False), flush=True)


def models() -> None:
    from monailabel.monai.vista_weights import pretrained_weights
    from monailabel.nninteractive import weights as nn
    from monailabel.providers import vista3d
    from monailabel.providers.sam import MODELS
    from monailabel.sam.weights import weights as sam_weights
    from monailabel.totalsegmentator.weights import manifest
    from monailabel.totalsegmentator.weights import pretrained_weights as total_weights

    root = Path(os.environ["MONAILABEL_MODELS_DIR"])
    fetch(
        vista3d.WEIGHT_URL,
        root / "vista3d" / vista3d.REVISION / "model.pt",
        "sha256",
        vista3d.CHECKSUM,
    )
    pretrained_weights()
    for provider, spec in MODELS.items():
        fetch(
            f"https://huggingface.co/{spec.repository}/resolve/{spec.revision}/{spec.filename}",
            root / provider / spec.revision / spec.filename,
            "sha256",
            spec.checksum,
        )
        sam_weights(provider)
    for name, (_, checksum) in nn.FILES.items():
        fetch(
            f"https://huggingface.co/MIC-DKFZ/nnInteractive/resolve/{nn.REVISION}/nnInteractive_v1.0/{name}",
            nn.cache_root() / nn.REVISION / name,
            "sha256",
            checksum,
        )
    nn.weights(lambda _: None)
    for provider in ("totalsegmentator-ct", "totalsegmentator-mr"):
        spec = manifest(provider)
        folder = root / "totalsegmentator" / spec["sha256"]
        if not (folder / spec["folder"]).exists():
            archive = fetch(
                spec["url"], root / "downloads" / f"{provider}.zip", "sha256", spec["sha256"]
            )
            folder.mkdir(parents=True, exist_ok=True)
            with (
                tempfile.TemporaryDirectory(dir=folder) as temp,
                zipfile.ZipFile(archive) as bundle,
            ):
                for name, checksum in spec["files"].items():
                    target = Path(temp) / name
                    if not target.resolve().is_relative_to(Path(temp).resolve()):
                        raise RuntimeError("Unsafe checkpoint path.")
                    content = bundle.read(name)
                    if hashlib.sha256(content).hexdigest() != checksum:
                        raise RuntimeError("Checkpoint entry checksum mismatch.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    target.chmod(0o444)
                (Path(temp) / spec["folder"]).rename(folder / spec["folder"])
        total_weights(provider)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models")
    samples = commands.add_parser("datasets")
    samples.add_argument("identifiers", nargs="+")
    viewers = commands.add_parser("viewer")
    viewers.add_argument("name", choices=("slicer", "qupath"))
    args = parser.parse_args()
    if args.command == "models":
        models()
    elif args.command == "datasets":
        datasets(args.identifiers)
    else:
        viewer(args.name)


if __name__ == "__main__":
    main()
