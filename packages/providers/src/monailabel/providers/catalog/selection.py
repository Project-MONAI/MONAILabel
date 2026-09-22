"""Keep the NVIDIA import picker focused on recent compatible model versions."""

import re

from monailabel.providers.catalog.discovery import Catalog


def family_version(identifier: str) -> tuple[str, tuple[int, int]] | None:
    model = identifier.rsplit("/", 1)[-1].removeprefix("bedrock-").removesuffix("-v1")
    model = re.sub(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$", "", model)
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?(?:[a-z]|-|$)", model)
    if match:
        return "gpt", (int(match[1]), int(match[2] or 0))
    match = re.match(r"o(\d+)(?:-|$)", model)
    if match:
        return "openai-o", (int(match[1]), 0)
    match = re.match(r"claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?(?:-|$)", model)
    if match:
        return "claude-" + match[1], (int(match[2]), int(match[3] or 0))
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?-(flash-lite|flash|pro)(?:-|$)", model)
    if match:
        return "gemini-" + match[3], (int(match[1]), int(match[2] or 0))
    return None


def recent_models(catalog: Catalog) -> Catalog:
    """Retain two versions per family, including their routes and size variants.

    Unversioned, capability-verified models remain discoverable. This picker
    policy does not constrain fixed presets, direct providers or existing imports.
    """
    versions: dict[str, set[tuple[int, int]]] = {}
    for model in catalog.models:
        version = family_version(model.id)
        if version:
            versions.setdefault(version[0], set()).add(version[1])
    latest = {family: set(sorted(values, reverse=True)[:2]) for family, values in versions.items()}
    selected = []
    for model in catalog.models:
        version = family_version(model.id)
        if version is None or version[1] in latest[version[0]]:
            selected.append(model)
    return Catalog(selected, catalog.excluded + len(catalog.models) - len(selected))
