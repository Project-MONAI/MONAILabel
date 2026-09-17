"""Synthetic fixtures for repeatable workflow checks; never claimed to be clinical data."""

import base64

import numpy as np

from monailabel.core.models import (
    ImageImport,
    Label,
    ModelRecord,
    ProjectCreate,
    Split,
)
from monailabel.server.data import Datasets, nifti_bytes
from monailabel.server.storage import Artifacts, Store


def create_demo(store: Store, artifacts: Artifacts, datasets: Datasets) -> dict[str, str]:
    project = datasets.create(
        ProjectCreate(
            name="Segmentation demo",
            instructions=(
                "Practice with synthetic sample images, not patient scans. "
                "Segment the two structures. Review all voxels and all labels."
            ),
            labels=[
                Label(id=0, name="Background", color="#18212f"),
                Label(id=1, name="Structure A", color="#32d6bb"),
                Label(id=2, name="Structure B", color="#f6ad55"),
            ],
        )
    )
    teacher = ModelRecord(
        project_id=project.id,
        name="Demo intensity thresholds",
        provider="threshold",
        label_ids=[0, 1, 2],
        config={"thresholds": [0.50, 0.89]},
    )
    with store.transaction() as session:
        session.insert(teacher)
        session.update(
            project.model_copy(
                update={
                    "is_demo": True,
                    "defaults": {1: teacher.id, 2: teacher.id},
                }
            )
        )
    affine: list[list[float]] = [[-1.2, 0, 0, 42], [0, 1.5, 0, -30], [0, 0, 2.0, 10], [0, 0, 0, 1]]
    x, y, z = np.indices((48, 40, 24))
    for index in range(12):
        rng = np.random.default_rng(1000 + index)
        shift = rng.integers(-2, 3, size=3)
        reference = np.zeros(x.shape, dtype=np.uint8)
        region_a = ((x - 16 - shift[0]) / 10) ** 2 + ((y - 19 - shift[1]) / 12) ** 2 + (
            (z - 12) / 8
        ) ** 2 < 1
        region_b = ((x - 34) / 7) ** 2 + ((y - 22) / 9) ** 2 + ((z - 11 - shift[2]) / 7) ** 2 < 1
        reference[region_a] = 1
        reference[region_b] = 2
        signal = np.asarray([0.08, 0.49, 0.84], dtype=np.float32)[reference]
        image = (signal + rng.normal(0, 0.035, x.shape)).astype(np.float32)
        split = Split.TRAIN if index < 6 else Split.VALIDATION if index < 9 else Split.POOL
        asset = datasets.import_image(
            project.id,
            ImageImport(
                name=f"synthetic-{index:02}.nii",
                group_id=f"synthetic-case-{index:02}",
                split=split,
                image_base64=base64.b64encode(nifti_bytes(image, affine)).decode(),
            ),
        )
        with store.transaction() as session:
            session.update(asset.model_copy(update={"fixture_key": artifacts.put_array(reference)}))
    return {"project_id": project.id, "baseline_id": teacher.id}
