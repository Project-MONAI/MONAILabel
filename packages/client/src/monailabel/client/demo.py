"""Exercise the complete HTTP workflow using explicitly synthetic fixture reviews."""

from collections.abc import Callable
from typing import Any

from monailabel.client.client import Client


def run_demo(client: Client, report: Callable[[str], None] = print) -> dict[str, Any]:
    setup = client.post("/api/demo")
    project_id, baseline = setup["project_id"], setup["baseline_id"]
    report(f"Created synthetic NIfTI project: {project_id}")
    report("Fixture masks simulate human corrections in this automated demo only.")
    assets = client.get(f"/api/projects/{project_id}/assets")
    for asset in assets:
        if asset["split"] == "pool":
            continue
        job = client.post(f"/api/assets/{asset['id']}/annotate", {"model_id": baseline})
        proposal_id = client.wait(job["id"])["proposal_id"]
        reference = client.get(f"/api/assets/{asset['id']}/fixture")["mask"]
        annotation = client.post(
            f"/api/assets/{asset['id']}/review",
            {
                "base_revision": 0,
                "proposal_id": proposal_id,
                "mask": reference,
                "covered_labels": [0, 1, 2],
                "reviewer": "synthetic-fixture-review",
            },
        )
        client.post(
            f"/api/annotations/{annotation['id']}/decision",
            {
                "verdict": "accepted",
                "comment": "Synthetic fixture verification only.",
            },
        )
    snapshot = client.post(f"/api/projects/{project_id}/snapshots")
    job = client.post(f"/api/projects/{project_id}/train", {"snapshot_id": snapshot["id"]})
    candidate = client.wait(job["id"])["model_id"]
    job = client.post(
        f"/api/projects/{project_id}/evaluate",
        {
            "snapshot_id": snapshot["id"],
            "candidate_id": candidate,
            "baseline_id": baseline,
        },
    )
    evaluation_id = client.wait(job["id"])["evaluation_id"]
    evaluation = client.get(f"/api/evaluations/{evaluation_id}")
    report(f"Baseline Dice: {evaluation['baseline']['mean_dice']:.4f}")
    report(f"Learned model Dice: {evaluation['candidate']['mean_dice']:.4f}")
    if not evaluation["eligible_labels"]:
        raise RuntimeError("Synthetic candidate did not meet promotion criteria.")
    project = client.get(f"/api/projects/{project_id}")
    promotion = client.post(
        f"/api/projects/{project_id}/promote",
        {
            "evaluation_id": evaluation_id,
            "label_ids": evaluation["eligible_labels"],
            "base_version": project["version"],
        },
    )
    pool = next(a for a in assets if a["split"] == "pool")
    job = client.post(f"/api/assets/{pool['id']}/annotate", {})
    proposal = client.get(f"/api/proposals/{client.wait(job['id'])['proposal_id']}")
    if proposal["model_ids"] != [candidate]:
        raise RuntimeError("Promoted model was not used for the next annotation.")
    job = client.post(
        f"/api/projects/{project_id}/select",
        {
            "strategy": "disagreement",
            "model_ids": [baseline, candidate],
            "limit": 2,
        },
    )
    selection = client.wait(job["id"])
    reference = client.get(f"/api/assets/{pool['id']}/fixture")["mask"]
    annotation = client.post(
        f"/api/assets/{pool['id']}/review",
        {
            "base_revision": 0,
            "proposal_id": proposal["id"],
            "mask": reference,
            "covered_labels": [0, 1, 2],
            "reviewer": "synthetic-fixture-review",
        },
    )
    client.post(
        f"/api/annotations/{annotation['id']}/decision",
        {
            "verdict": "accepted",
            "comment": "Synthetic fixture verification only.",
        },
    )
    client.post(f"/api/assets/{pool['id']}/assign-train")
    next_snapshot = client.post(f"/api/projects/{project_id}/snapshots")
    job = client.post(
        f"/api/projects/{project_id}/train",
        {
            "snapshot_id": next_snapshot["id"],
            "mode": "continue",
            "parent_model_id": candidate,
            "name": "Second learning round",
        },
    )
    second = client.wait(job["id"])
    report("Verified: reviewed masks → real training → held-out evaluation → model handoff.")
    report("Verified: active selection and a second training round using one newly reviewed case.")
    return {
        **setup,
        "snapshot_id": snapshot["id"],
        "model_id": candidate,
        "evaluation_id": evaluation_id,
        "promotion_id": promotion["id"],
        "second_model_id": second["model_id"],
        "second_training_samples": second["training_samples"],
        "selection": selection,
        "asset_id": pool["id"],
    }
