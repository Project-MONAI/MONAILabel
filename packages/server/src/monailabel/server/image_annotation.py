"""Tiled 2D inference independent of viewer objects and model vendors."""

import numpy as np

from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import AnnotateRequest, Project
from monailabel.core.ports import Image, Mask
from monailabel.core.tiling import ImageTile
from monailabel.server.jobs import JobContext
from monailabel.server.models import Models


def annotate_tiles(
    image: Image,
    mask: Mask,
    project: Project,
    assignments: dict[str, list[int]],
    label_ids: list[int],
    request: AnnotateRequest,
    tiles: list[ImageTile],
    models: Models,
    context: JobContext,
) -> None:
    steps = len(tiles) * len(assignments)
    completed = 0
    for index, tile in enumerate(tiles):
        r, w = tile.read, tile.write
        read = (slice(r.y, r.y + r.height), slice(r.x, r.x + r.width))
        write = (slice(w.y, w.y + w.height), slice(w.x, w.x + w.width))
        owned = (slice(w.y - r.y, w.y - r.y + w.height), slice(w.x - r.x, w.x - r.x + w.width))
        target = mask[write]
        target[np.isin(target, label_ids)] = 0
        supplied = image[read]
        for identifier, selected_labels in assignments.items():
            model = models.for_labels(models.get(project.id, identifier), selected_labels)
            detail = f"Tile {index + 1} of {len(tiles)} · {model.name}"
            context.progress(completed / steps, detail)
            prompt = request.prompt + (
                "\nThis is one native-resolution image tile. Segment only visible targets; "
                "return coordinates relative to this tile. Do not outline the tile border. "
            )
            try:
                result = models.predict(project, model, supplied, prompt)[owned]
            except DomainError as exc:
                raise DomainError(f"{detail}: {exc}", code=exc.code, status=exc.status) from exc
            selected = np.isin(result, selected_labels)
            if np.any(selected & (target != 0) & (target != result)):
                raise Conflict("Tile predictions overlap another preserved annotation class.")
            target[selected] = result[selected]
            completed += 1
            context.progress(completed / steps, detail)
