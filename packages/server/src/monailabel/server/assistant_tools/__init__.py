"""The coordinator's finite tool catalog, shared by every client."""

from . import annotation, datasets, evaluation, learning, spatial, workspace
from .base import ToolContext, ToolRegistry


def catalog(context: ToolContext) -> ToolRegistry:
    registry = ToolRegistry(context)
    workspace.register(registry)
    datasets.register(registry)
    annotation.register(registry)
    spatial.register(registry)
    learning.register(registry)
    evaluation.register(registry)
    return registry
