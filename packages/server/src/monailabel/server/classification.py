"""Immutable classification proposals for existing annotation objects."""

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    Asset,
    ClassificationProposal,
    ClassificationRequest,
    Job,
    Project,
)
from monailabel.server.jobs import JobContext, Jobs, Outcome
from monailabel.server.models import Models
from monailabel.server.storage import Artifacts, Store


class Classifications:
    def __init__(self, store: Store, artifacts: Artifacts, models: Models, jobs: Jobs):
        self.store, self.artifacts, self.models, self.jobs = store, artifacts, models, jobs

    def classify(self, asset_id: str, request: ClassificationRequest) -> Job:
        asset = self.store.get(Asset, asset_id)
        if asset.kind != "image2d":
            raise DomainError("Object classification currently requires a 2D image in QuPath.")
        if asset.revision != request.base_revision:
            raise Conflict("Annotation revision changed. Reload before classification.")
        project = self.store.get(Project, asset.project_id)
        source_ids = {label.id for label in project.labels if label.id}
        if any(o.label_id not in source_ids for o in request.objects):
            raise DomainError("Classify objects with project foreground labels.")
        if any(
            o.region.x + o.region.width > asset.spatial_shape[1]
            or o.region.y + o.region.height > asset.spatial_shape[0]
            for o in request.objects
        ):
            raise DomainError("Classification objects must lie inside the source image.")
        if sum(o.region.width * o.region.height for o in request.objects) > 16_777_216:
            raise DomainError("The selected classification objects exceed the image pixel limit.")
        identifier = request.model_id
        if not identifier:
            defaults = {project.defaults.get(o.label_id) for o in request.objects}
            if len(defaults) == 1:
                identifier = defaults.pop()
        if not identifier:
            raise DomainError("Select a classification-capable vision model.")
        model = self.models.get(project.id, identifier)
        if model.provider not in self.models.classifiers:
            raise DomainError(
                "This model has no object-classification adapter. "
                "Select a configured vision API model."
            )
        x, y = min(o.region.x for o in request.objects), min(o.region.y for o in request.objects)
        right = max(o.region.x + o.region.width for o in request.objects)
        bottom = max(o.region.y + o.region.height for o in request.objects)
        local = [
            o.model_copy(
                update={
                    "region": o.region.model_copy(update={"x": o.region.x - x, "y": o.region.y - y})
                }
            )
            for o in request.objects
        ]

        def work(context: JobContext) -> Outcome:
            context.progress(0, f"Classifying {len(local)} objects · {model.name}")
            image = self.artifacts.array(asset.image_key)[y:bottom, x:right]
            results = self.models.classifiers[model.provider].classify(
                image, local, request.categories, request.prompt, model
            )
            expected = {o.id for o in request.objects}
            if (
                len(results) != len(expected)
                or {r.object_id for r in results} != expected
                or any(
                    r.category is not None and r.category not in request.categories for r in results
                )
            ):
                raise DomainError(
                    "Classifier returned missing, duplicate, or invalid object categories."
                )
            context.progress(1, "Classification proposals ready for review")
            proposal = ClassificationProposal(
                project_id=project.id,
                asset_id=asset.id,
                base_revision=asset.revision,
                model_id=model.id,
                request=request,
                results=results,
            )
            return Outcome({"classification_id": proposal.id}, [proposal])

        return self.jobs.submit(
            "classify", project.id, request.model_dump(mode="json") | {"asset_id": asset.id}, work
        )
