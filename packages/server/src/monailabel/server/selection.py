"""Explicit active-learning strategies operating only on the annotation pool."""

import random

import numpy as np

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset, Job, Project, SelectionRequest, Split, WorkItem
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.models import Models
from monailabel.server.storage import Artifacts, Store


class Selection:
    def __init__(self, store: Store, artifacts: Artifacts, models: Models, jobs: Jobs):
        self.store, self.artifacts, self.models, self.jobs = store, artifacts, models, jobs

    def select(self, project_id: str, request: SelectionRequest) -> Job:
        project = self.store.get(Project, project_id)
        candidates = [
            a
            for a in self.store.list(Asset, project_id)
            if a.split == Split.POOL and a.annotation_id is None
        ]
        model_records = [self.models.get(project_id, x) for x in request.model_ids]
        if request.strategy == "disagreement":
            if len(model_records) != 2 or model_records[0].id == model_records[1].id:
                raise DomainError("Disagreement selection requires two distinct models.")
            for index, model in enumerate(model_records):
                if self.models.promptable(model):
                    other = model_records[1 - index]
                    labels = [i for i in other.label_ids if i] or [
                        label.id for label in project.labels if label.id
                    ]
                    model_records[index] = self.models.for_labels(model, labels)
            if set(model_records[0].label_ids) != set(model_records[1].label_ids):
                raise DomainError("Disagreement models must use the same class coverage.")

        def work(context: JobContext) -> Outcome:
            items = []
            rng = random.Random(request.seed)
            for index, asset in enumerate(candidates):
                context.progress(index / max(1, len(candidates)))
                if request.strategy == "random":
                    score = rng.random()
                else:
                    image = self.artifacts.array(asset.image_key)
                    predictions = [
                        self.models.predict(project, m, image, affine=asset.affine)
                        for m in model_records
                    ]
                    score = float(np.mean(predictions[0] != predictions[1]))
                items.append(WorkItem(asset_id=asset.id, score=score, reason=request.strategy))
            items.sort(key=lambda x: (-x.score, x.asset_id))
            seen_groups: set[str] = set()
            groups = {a.id: a.group_id for a in candidates}
            selected = []
            for item in items:
                if groups[item.asset_id] in seen_groups:
                    continue
                seen_groups.add(groups[item.asset_id])
                selected.append(item)
                if len(selected) == request.limit:
                    break
            return Outcome({"items": [x.model_dump(mode="json") for x in selected]})

        return self.jobs.submit("select", project_id, request.model_dump(mode="json"), work)
