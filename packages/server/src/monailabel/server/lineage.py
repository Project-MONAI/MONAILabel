"""Training identity across immutable checkpoint ancestry, including image aliases."""

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, Snapshot, Split
from monailabel.server.storage import Store


def training_identity(store: Store, model: ModelRecord) -> tuple[set[str], set[str]]:
    groups: set[str] = set()
    images: set[str] = set()
    visited: set[str] = set()
    with store.transaction() as session:
        current = model
        while True:
            if current.id in visited or current.project_id != model.project_id:
                raise DomainError("Model ancestry is inconsistent; evaluation cannot proceed.")
            visited.add(current.id)
            groups.update(current.training_groups)
            if current.snapshot_id:
                snapshot = session.get(Snapshot, current.snapshot_id)
                images.update(
                    sample.image_key
                    for sample in snapshot.samples
                    if sample.split == Split.TRAIN and sample.asset_id in current.training_assets
                )
            if not current.parent_id:
                break
            current = session.get(ModelRecord, current.parent_id)
    return groups, images
