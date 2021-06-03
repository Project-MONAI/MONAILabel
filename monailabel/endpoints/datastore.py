import json
import logging
import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from monailabel.interfaces import Datastore, DefaultLabelTag, MONAILabelApp
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

    logger.debug(f"output type: {output}")
    if output == ResultType.all:
        return json.loads(str(d))
    if output == ResultType.train:
        return d.datalist()
    return d.status()


@router.put("/", summary="Upload new Image")
async def add_image(image: str, file: UploadFile = File(...)):
    logger.info(f"Image: {image}; File: {file}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")


@router.delete("/", summary="Remove Image/Label")
async def remove_image(id: str):
    logger.info(f"Image/Label: {id}")
    raise HTTPException(status_code=501, detail="Not Implemented Yet")


@router.put("/label", summary="Save Finished Label")
async def save_label(image: str, tag: str = DefaultLabelTag.FINAL.value, label: UploadFile = File(...)):
    file_ext = "".join(pathlib.Path(label.filename).suffixes) if label.filename else ".nii.gz"
    label_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
    tag = tag if tag else DefaultLabelTag.FINAL.value

    with open(label_file, "wb") as buffer:
        shutil.copyfileobj(label.file, buffer)

    instance: MONAILabelApp = get_app_instance()
    label_id = instance.datastore().save_label(image, label_file, tag)

    res = instance.on_save_label(image, label_id)
    res = res if res else {}
    res.update(
        {
            "image": image,
            "label": label_id,
        }
    )
    return res
