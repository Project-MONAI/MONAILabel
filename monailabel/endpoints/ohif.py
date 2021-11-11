import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from monailabel.utils.others.generic import get_mime_type

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ohif",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


def get_ohif(path: str):
    ohif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static", "ohif")
    file = os.path.join(ohif_dir, "index.html")
    if path:
        path = os.path.join(ohif_dir, path.replace("/", os.pathsep))
        file = path if os.path.exists(path) else file

    if not os.path.exists(file):
        logger.info(file)
        raise HTTPException(status_code=404, detail="Resource NOT Found")
    return FileResponse(file, media_type=get_mime_type(file))


@router.get("/{path:path}", include_in_schema=False)
async def api_get_ohif(path: str):
    return get_ohif(path)
