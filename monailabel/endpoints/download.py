import base64
import os
from math import ceil

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from interfaces import Datastore
from monailabel.utils.others.generic import get_mime_type
from utils.datastore import LocalDatastore

router = APIRouter(
    prefix="/download",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.get("/{image_id}", summary="Download Image")
async def download(image_id):

    datastore: Datastore = LocalDatastore(settings.STUDIES)
    _, image_path = datastore.get_image(image_id=image_id)

    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image NOT Found")
    return FileResponse(image_path, media_type=get_mime_type(image_path), filename=image_id)
