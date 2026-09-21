"""Persistent browser desktop identity, independent of its runtime and HTTP transport."""

from typing import Literal

from monailabel.core.models import Record


class DesktopSession(Record):
    project_id: str
    user_id: str
    asset_id: str
    viewer: Literal["slicer", "qupath"]
    mode: Literal["annotation", "review"]
    credential_id: str
    ended: bool = False
