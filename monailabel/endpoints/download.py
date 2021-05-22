import base64
import os
from math import ceil

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from monailabel.utils.others.generic import get_mime_type

router = APIRouter(
    prefix="/download",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.get("/{image}", summary="Download Image")
async def download(image):
    image = image.ljust(ceil(len(image) / 4) * 4, "=")
    image = base64.urlsafe_b64decode(image.encode("utf-8")).decode("utf-8")
    if not os.path.isfile(image):
        raise HTTPException(status_code=404, detail="Image NOT Found")
    return FileResponse(image, media_type=get_mime_type(image), filename=os.path.basename(image))
