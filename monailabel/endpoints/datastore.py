import json
import logging
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile

from monailabel.interfaces import Datastore
from monailabel.utils.others.generic import get_app_instance

logger = logging.getLogger(__name__)
train_tasks: List = []
train_process: Dict = dict()

router = APIRouter(
    prefix="/datastore",
    tags=["Datastore"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get All Images/Labels from datastore")
async def datastore(train: bool = True):
    d: Datastore = get_app_instance().datastore()
    return d.datalist() if train else json.loads(str(d))


@router.put("/", summary="Upload new Image")
async def add_image(image: str, file: UploadFile = File(...)):
    logger.info(f"Image: {image}; File: {file}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")


@router.delete("/", summary="Remove Image/Label")
async def remove_image(image: str):
    logger.info(f"Image: {image}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")
