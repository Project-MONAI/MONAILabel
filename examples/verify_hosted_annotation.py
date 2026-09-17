"""Run one explicit hosted annotation request and measure only its requested slice.

Uses the normal authenticated API. The environment API key is resolved by the
server; the CLI session is separate. No proposal is submitted or approved here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from monailabel.client.client import Client
from monailabel.viewers.resources.geometry import source_slice


def verify(
    client: Client,
    project_id: str,
    asset_id: str,
    *,
    axis: int,
    index: int,
    window: list[float],
    config: Path,
    model_id: str | None = None,
    reference_annotation_id: str | None = None,
    radiological: bool = False,
) -> dict[str, Any]:
    asset = client.get(f"/api/assets/{asset_id}")
    if asset["project_id"] != project_id:
        raise ValueError("The sample belongs to a different project.")
    if model_id is None:
        model = client.post(f"/api/projects/{project_id}/models", json.loads(config.read_text()))
        model_id = model["id"]
    else:
        model = next(
            m for m in client.get(f"/api/projects/{project_id}/models") if m["id"] == model_id
        )
    scope = {"axis": axis, "index": index, "window": window}
    if radiological:
        if axis != 2:
            raise ValueError("The radiological example currently handles source axis 2 only.")
        affine = np.asarray(asset["affine"])
        # Conventional axial display: columns towards patient left, rows posterior.
        xy_to_ras = np.diag([-1.0, 1.0, 1.0, 1.0])
        center = (np.asarray(asset["spatial_shape"]) - 1) / 2
        center[axis] = index
        xy_to_ras[:3, 3] = (affine @ np.append(center, 1))[:3]
        scope = source_slice(np.linalg.inv(affine), xy_to_ras, asset["spatial_shape"], window)
        if scope["axis"] != axis:
            raise ValueError("This volume's source axis 2 is not an axial plane.")
    project = client.get(f"/api/projects/{project_id}")
    labels = [
        label for label in project["labels"] if label["id"] in model["label_ids"] and label["id"]
    ]
    job = client.post(
        f"/api/assets/{asset_id}/annotate",
        {
            "model_id": model_id,
            "label_ids": [label["id"] for label in labels],
            "slice": scope,
            "prompt": "Segment "
            + ", ".join(label["name"] for label in labels)
            + " in this CT slice. Trace the anatomical boundary as a polygon for human review.",
        },
    )
    print(f"Annotation job: {job['id']}", flush=True)
    proposal_id = client.wait(job["id"], timeout=300)["proposal_id"]
    proposal = client.get(f"/api/proposals/{proposal_id}")
    values = np.asarray(client.get(f"/api/proposals/{proposal_id}/mask")["mask"], dtype=np.uint8)
    region: list[slice | int] = [slice(None)] * 3
    region[axis] = index
    predicted = values[tuple(region)]
    report = {
        "project_id": project_id,
        "asset_id": asset_id,
        "asset_name": asset["name"],
        "model_id": model_id,
        "provider": model["provider"],
        "provider_model": model["config"].get("model"),
        "proposal_id": proposal_id,
        "proposal_status": proposal["status"],
        "base_revision": proposal["base_revision"],
        "slice": scope,
        "predicted_pixels": int(np.count_nonzero(predicted)),
        "submitted": False,
        "reviewer_accepted": False,
    }
    if reference_annotation_id:
        history = client.get(f"/api/assets/{asset_id}/annotations")
        if not any(a["id"] == reference_annotation_id for a in history):
            raise ValueError("Reference annotation does not belong to this sample.")
        reference = np.asarray(
            client.get(f"/api/annotations/{reference_annotation_id}/mask")["mask"], dtype=np.uint8
        )
        reference_slice = reference[tuple(region)]
        scores = {}
        for label in labels:
            truth, output = reference_slice == label["id"], predicted == label["id"]
            denominator = int(truth.sum() + output.sum())
            scores[label["name"]] = (
                2 * int(np.count_nonzero(truth & output)) / denominator if denominator else None
            )
        report.update(
            {
                "reference_annotation_id": reference_annotation_id,
                "reference_pixels": int(np.count_nonzero(reference_slice)),
                "slice_dice": scores,
                "comparison_scope": "Only the requested 2D slice; not a volume benchmark.",
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_id")
    parser.add_argument("asset_id")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--window", nargs=2, type=float, default=[-160, 240])
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "models/nvidia-sol.json"
    )
    parser.add_argument("--model-id", help="Reuse an already registered model")
    parser.add_argument(
        "--reference-annotation-id", help="Compare against this saved reference revision"
    )
    parser.add_argument("--output", type=Path, help="Save a credential-free JSON report")
    parser.add_argument(
        "--radiological",
        action="store_true",
        help="Use standard axial display orientation, mapped through the source affine",
    )
    args = parser.parse_args()
    with Client(args.url) as client:
        result = verify(
            client,
            args.project_id,
            args.asset_id,
            axis=args.axis,
            index=args.index,
            window=args.window,
            config=args.config,
            model_id=args.model_id,
            reference_annotation_id=args.reference_annotation_id,
            radiological=args.radiological,
        )
    content = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
