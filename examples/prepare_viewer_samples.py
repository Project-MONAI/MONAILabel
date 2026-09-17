"""Import public pathology or derived DICOM samples through the public API.

Run `uv run monailabel login --username ...` first. No inference, review, or
training is performed. Repeated runs reuse downloaded files, projects and series.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
import pydicom
from PIL import Image

from monailabel.client.client import Client
from monailabel.dicom.series import derived_ct_series

PATHOLOGY_URL = (
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
)
PATHOLOGY_SHA256 = "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7"


def project(client, name, label):
    existing = next((p for p in client.get("/api/projects") if p["name"] == name), None)
    return existing or client.post(
        "/api/projects",
        {
            "name": name,
            "labels": [
                {"id": 0, "name": "Background", "color": "#000000"},
                {"id": 1, "name": label, "color": "#22c55e"},
            ],
        },
    )


def configure_models(client, project_id, max_output_tokens=4096, *, make_sol_default=False):
    path = "/api/projects/" + project_id
    existing = client.get(path + "/models")
    for filename in ["nvidia-sol.json", "nvidia-astra.json"]:
        body = json.loads((Path(__file__).parent / "models" / filename).read_text())
        body["config"]["max_output_tokens"] = max_output_tokens
        model = next(
            (m for m in existing if m["config"].get("model") == body["config"]["model"]), None
        )
        model = model or client.post(path + "/models", body)
        if filename == "nvidia-sol.json" and make_sol_default:
            client.request(
                "PUT",
                path + "/annotation-model",
                {
                    "model_id": model["id"],
                    "base_version": client.get(path)["version"],
                },
            )


def pathology(client, directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "CMU-1-Small-Region.svs"
    if not source.exists():
        response = httpx.get(PATHOLOGY_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != PATHOLOGY_SHA256:
            raise RuntimeError("Public pathology sample checksum differs from the verified file.")
        source.write_bytes(response.content)
    if hashlib.sha256(source.read_bytes()).hexdigest() != PATHOLOGY_SHA256:
        raise RuntimeError("The cached pathology sample has changed.")
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        crops = []
        for index, (x, y) in enumerate(
            [(900, 650), (1200, 650), (900, 1150), (1200, 1150), (900, 1700), (1200, 1700)], 1
        ):
            path = directory / f"CMU-1-nuclei-{index:02d}.png"
            rgb.crop((x, y, x + 256, y + 256)).save(path)
            crops.append({"file": path.name, "source_xy": [x, y], "size": [256, 256]})
    p = project(client, "Pathology · CMU nuclei", "Nuclei")
    assets = []
    for path in [source, *(directory / c["file"] for c in crops)]:
        response = client.http.post(
            f"/api/projects/{p['id']}/assets/upload",
            params={
                "name": path.name,
                "group_id": "OpenSlide-CMU-1",
                "split": "pool",
            },
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        assets.append(response.json()["id"])
    configure_models(client, p["id"], max_output_tokens=16384, make_sol_default=True)
    manifest = {
        "project_id": p["id"],
        "asset_ids": assets,
        "source_url": PATHOLOGY_URL,
        "sha256": PATHOLOGY_SHA256,
        "license": "CC0 1.0",
        "crops": crops,
        "notes": "One source slide; no independent validation set or expert labels.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def dicom(client, directory, nifti_directory, orthanc_url):
    directory.mkdir(parents=True, exist_ok=True)
    p = project(client, "Radiology · Spleen DICOM", "Spleen")
    records = []
    for name in ["spleen_3", "spleen_8"]:
        folder = directory / name
        folder.mkdir(exist_ok=True)
        metadata = folder / "series.json"
        if not metadata.exists():
            source = nifti_directory / (name + ".nii.gz")
            nii = nib.load(source)
            contents = derived_ct_series(nii.get_fdata(dtype=np.float32), nii.affine, name)
            for index, content in enumerate(contents, 1):
                (folder / f"{index:04d}.dcm").write_bytes(content)
            first = pydicom.dcmread(io.BytesIO(contents[0]))
            metadata.write_text(
                json.dumps(
                    {
                        "name": name,
                        "source": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "study_uid": str(first.StudyInstanceUID),
                        "series_uid": str(first.SeriesInstanceUID),
                        "metadata": "Derived CT: synthetic patient and acquisition identifiers.",
                    },
                    indent=2,
                )
                + "\n"
            )
        record = json.loads(metadata.read_text())
        with httpx.Client(base_url=orthanc_url.rstrip("/"), timeout=120) as orthanc:
            for path in sorted(folder.glob("*.dcm")):
                response = orthanc.post("/instances", content=path.read_bytes())
                response.raise_for_status()
        job = client.post(
            f"/api/projects/{p['id']}/dicom-series", {"series_uid": record["series_uid"]}
        )
        record.update(client.wait(job["id"]))
        records.append(record)
    configure_models(client, p["id"])
    manifest = {
        "project_id": p["id"],
        "series": records,
        "notes": "Derived Decathlon CT test data; no annotations accepted or copied.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["pathology", "dicom"])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--directory", type=Path)
    parser.add_argument(
        "--nifti-directory",
        type=Path,
        default=Path.home() / "Downloads/Task09_Spleen/Task09_Spleen/imagesTr",
    )
    parser.add_argument("--orthanc-url", default="http://127.0.0.1:8042")
    args = parser.parse_args()
    directory = args.directory or Path.home() / "Downloads" / (
        "MONAILabel-Pathology-CMU" if args.kind == "pathology" else "MONAILabel-Spleen-DICOM"
    )
    with Client(args.url) as client:
        result = (
            pathology(client, directory)
            if args.kind == "pathology"
            else dicom(client, directory, args.nifti_directory, args.orthanc_url)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
