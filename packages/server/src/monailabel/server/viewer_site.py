"""Serve the built OHIF application on the same origin as authenticated APIs."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from monailabel.core.errors import DomainError
from monailabel.viewers.ohif import OhifManager

router = APIRouter()


@router.get("/ohif/{path:path}")
def ohif(path: str) -> FileResponse:
    root = OhifManager().dist.resolve()
    if not (root / "index.html").exists():
        raise DomainError("Prepare OHIF first: uv run monailabel viewer ohif", status=503)
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise DomainError("Unknown viewer resource.", status=404)
    if not candidate.is_file():
        if candidate.suffix:
            raise DomainError("Unknown viewer resource.", status=404)
        candidate = root / "index.html"
    return FileResponse(candidate)
