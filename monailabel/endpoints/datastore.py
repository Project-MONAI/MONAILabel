import json
import logging
from enum import Enum
from typing import Dict, List, Optional

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


class ResultType(str, Enum):
    train = "train"
    stats = "stats"
    all = "all"


@router.get("/", summary="Get All Images/Labels from datastore")
async def datastore(output: Optional[ResultType] = None):
    d: Datastore = get_app_instance().datastore()
    output = output if output else ResultType.stats
    logger.info(f"output type: {output}")
    if output == ResultType.all:
        return json.loads(str(d))
    if output == ResultType.train:
        return d.datalist()
    return {"total": len(d.list_images()), "completed": len(d.get_labeled_images()), "train": d.datalist()}


@router.put("/", summary="Upload new Image")
async def add_image(image: str, file: UploadFile = File(...)):
    logger.info(f"Image: {image}; File: {file}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")


@router.delete("/", summary="Remove Image/Label")
async def remove_image(image: str):
    logger.info(f"Image: {image}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")
