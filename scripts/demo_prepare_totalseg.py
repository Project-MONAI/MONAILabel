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

"""Prepare real all-structure CT predictions from already cached Decathlon archives.

Opt-in host operation, never a visitor tool. One GPU worker at a time; no downloads,
workspace mutations, reference masks or clinical accuracy claims. Resume skips only
checksum-verified completed predictions. Keep outputs on host-owned storage.
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from monailabel.core.ports import Volume
from monailabel.totalsegmentator import prepared
from monailabel.totalsegmentator.catalog import VERSION, targets
from monailabel.totalsegmentator.config import TotalConfig
from monailabel.totalsegmentator.runtime import run_worker
from monailabel.totalsegmentator.weights import manifest, pretrained_weights


def idle(database: Path | None) -> bool:
    if database is None:
        return True
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for kind, data in db.execute(
            "SELECT kind,data FROM records WHERE kind IN ('Job','DesktopSession')"
        ):
            item = json.loads(data)
            if kind == "Job" and item["status"] not in {
                "succeeded",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return False
            if kind == "DesktopSession" and not item.get("ended"):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-database", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    spec, config = manifest("totalsegmentator-ct"), TotalConfig()
    # Check before calling the normal verifier: this command cannot populate missing weights.
    base = args.models / "totalsegmentator" / spec["sha256"] / spec["folder"]
    if not base.is_dir():
        raise SystemExit("CT weights must already be cached; no downloads are allowed.")
    import os

    os.environ["MONAILABEL_MODELS_DIR"] = str(args.models)
    assert pretrained_weights("totalsegmentator-ct") == base
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for name in ("Task09_Spleen", "Task06_Lung", "Task03_Liver"):
        archive_path = args.datasets / (name + ".tar")
        if not archive_path.is_file():
            continue
        with tarfile.open(archive_path) as archive:
            members = sorted(
                (
                    m
                    for m in archive
                    if m.isfile()
                    and ("/imagesTr/" in m.name or "/imagesTs/" in m.name)
                    and m.name.endswith(".nii.gz")
                    and "/._" not in m.name
                ),
                key=lambda m: m.name,
            )
            for member in members:
                if args.limit and completed >= args.limit:
                    return
                while not idle(args.idle_database):
                    print("Waiting for private application jobs/desktops to finish", flush=True)
                    time.sleep(30)
                started = time.monotonic()
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="prepare-", dir=args.output
                    ) as temporary:
                        work = Path(temporary)
                        source_stream = archive.extractfile(member)
                        assert source_stream is not None
                        with source_stream, (work / "image.nii.gz").open("wb") as target:
                            shutil.copyfileobj(source_stream, target)
                        source = nib.load(work / "image.nii.gz")
                        volume = Volume(
                            np.asarray(source.dataobj, dtype=np.float32)[..., None],
                            np.asarray(source.affine, dtype=float).tolist(),
                        )
                        key = prepared.identity("totalsegmentator-ct", volume, config)
                        if (
                            prepared.read(args.output, key, "totalsegmentator-ct", source.shape)
                            is not None
                        ):
                            completed += 1
                            print(
                                json.dumps(
                                    {
                                        "source": member.name,
                                        "status": "already_prepared",
                                        "completed": completed,
                                    }
                                ),
                                flush=True,
                            )
                            continue
                        run_worker(
                            work,
                            {
                                "operation": "predict",
                                "provider": "totalsegmentator-ct",
                                "base": str(base),
                                "config": config.model_dump(mode="json"),
                            },
                            lambda _: None,
                        )
                        result = nib.load(work / "mask.nii.gz")
                        values = np.asarray(result.dataobj)
                        if values.shape != source.shape or not np.allclose(
                            result.affine, source.affine, rtol=1e-5, atol=1e-4
                        ):
                            raise ValueError("Source geometry changed")
                        if not set(np.unique(values)) <= {
                            0,
                            *targets("totalsegmentator-ct").values(),
                        }:
                            raise ValueError("Invalid class IDs")
                        destination = args.output / key
                        destination.mkdir(exist_ok=True)
                        np.save(
                            destination / "mask.npy", values.astype(np.uint8), allow_pickle=False
                        )
                        with (destination / "mask.npy").open("rb") as stream:
                            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                        evidence = {
                            "identity": key,
                            "source": member.name,
                            "mask_sha256": checksum,
                            "seconds": round(time.monotonic() - started, 2),
                            "geometry_preserved": True,
                            "shape": source.shape,
                            "affine": volume.affine,
                            "supported_structures": 117,
                            "present_structures": len(set(np.unique(values)) - {0}),
                            "version": VERSION,
                            "weights_sha256": spec["sha256"],
                            "resolution": "fast 3mm",
                            "clinical_accuracy_evaluated": False,
                        }
                        (destination / "prediction.json").write_text(json.dumps(evidence, indent=2))
                        completed += 1
                        print(json.dumps(evidence | {"completed": completed}), flush=True)
                except Exception as error:
                    print(
                        json.dumps(
                            {"source": member.name, "status": "failed", "error": str(error)}
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
