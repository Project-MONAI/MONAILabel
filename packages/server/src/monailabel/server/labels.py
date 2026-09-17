"""Append-only project labels resolved from explicit annotation targets."""

from monailabel.core.colors import default_color, normalize
from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import Label, LabelColorsUpdate, ModelRecord, Project
from monailabel.server.models import Models
from monailabel.server.storage import Store


def imported_labels(project: Project, names: dict[int, str]) -> tuple[Project, dict[int, int]]:
    """Map external label values by structure name without changing existing identities."""
    normalized = [normalize(name) for name in names.values()]
    if len(set(normalized)) != len(normalized):
        raise DomainError("Dataset contains ambiguous structure names.")
    labels = list(project.labels)
    by_name = {normalize(label.name): label.id for label in labels}
    mapping = {}
    for original, name in names.items():
        clean = normalize(name)
        if clean in by_name and by_name[clean] == 0:
            raise DomainError("Foreground structures cannot use the background label.")
        if clean not in by_name:
            identifier = max(label.id for label in labels) + 1
            if len(labels) >= 32 or identifier > 255:
                raise DomainError(
                    "This import would exceed the project's 31 foreground label limit."
                )
            labels.append(Label(id=identifier, name=clean.capitalize()))
            by_name[clean] = identifier
        mapping[original] = by_name[clean]
    if labels != project.labels:
        project = project.model_copy(update={"labels": labels, "version": project.version + 1})
    return project, mapping


def prompt_model(store: Store, project: Project, identifier: str | None) -> str | None:
    if identifier:
        return identifier
    if project.annotation_model_id:
        return project.annotation_model_id
    defaults = set(project.defaults.values())
    if len(defaults) == 1:
        return defaults.pop()
    if not defaults:
        models = [
            m
            for m in store.list(ModelRecord, project.id)
            if not m.archived and Models.promptable(m)
        ]
        if len(models) == 1:
            return models[0].id
    return None


def resolve_labels(
    store: Store,
    project_id: str,
    names: list[str],
    model_id: str | None,
    *,
    set_defaults: bool = True,
) -> tuple[Project, list[int]]:
    with store.transaction() as session:
        project = session.get(Project, project_id)
        existing = {label.name.casefold(): label for label in project.labels}
        missing = [name for name in names if name.casefold() not in existing]
        if missing:
            if not model_id:
                raise DomainError(
                    "Select a promptable annotation model before adding a new target."
                )
            model = session.get(ModelRecord, model_id)
            if model.archived:
                raise DomainError("This model has been deleted from the active catalog.")
            if model.project_id != project_id:
                raise DomainError("Model belongs to another project.")
            Models.validate_targets(model, missing)
            if len(project.labels) + len(missing) > 32:
                raise DomainError("This project has reached its 32-label limit.")
            labels = list(project.labels)
            defaults = dict(project.defaults)
            for name in missing:
                identifier = max(label.id for label in labels) + 1
                if identifier > 255:
                    raise DomainError("No uint8 label IDs remain in this project.")
                label = Label(
                    id=identifier,
                    name=name[:1].upper() + name[1:],
                )
                labels.append(label)
                existing[name.casefold()] = label
                if set_defaults:
                    defaults[identifier] = model_id
            project = project.model_copy(
                update={"labels": labels, "defaults": defaults, "version": project.version + 1}
            )
            session.update(project)
        return project, [existing[name.casefold()].id for name in names]


def update_colors(store: Store, project_id: str, request: LabelColorsUpdate) -> Project:
    """Change shared presentation only; masks and label/protocol identities stay intact."""
    with store.transaction() as session:
        project = session.get(Project, project_id)
        if request.base_version != project.version:
            raise Conflict("Project changed. Refresh before updating label colors.")
        targets = {normalize(name) for name in request.targets}
        labels = {normalize(label.name): label for label in project.labels if label.id}
        if not targets <= labels.keys():
            raise DomainError("Choose existing foreground labels to change their colors.")
        updated = [
            label.model_copy(
                update={
                    "color": request.color.lower()
                    if request.color
                    else default_color(label.name, label.id)
                }
            )
            if normalize(label.name) in targets
            else label
            for label in project.labels
        ]
        if updated == project.labels:
            return project
        project = project.model_copy(update={"labels": updated, "version": project.version + 1})
        session.update(project)
        return project
