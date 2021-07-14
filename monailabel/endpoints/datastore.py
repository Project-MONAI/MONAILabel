import json
import logging
import os
import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks

from monailabel.interfaces import Datastore, DefaultLabelTag, MONAILabelApp
from monailabel.utils.others.generic import get_app_instance, get_mime_type, remove_file

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


class RemoveType(str, Enum):
    image = "image"
    label = "label"
    label_tag = "label_tag"


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
async def add_image(background_tasks: BackgroundTasks, image: Optional[str] = None, file: UploadFile = File(...)):
    logger.info(f"Image: {image}; File: {file}")
    file_ext = "".join(pathlib.Path(file.filename).suffixes) if file.filename else ".nii.gz"

    image_id = image if image else os.path.basename(file.filename)
    image_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

    with open(image_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(remove_file, image_file)

    instance: MONAILabelApp = get_app_instance()
    image_id = instance.datastore().add_image(image_id, image_file)
    return {"image": image_id}


@router.delete("/", summary="Remove Image/Label")
async def remove_image(id: str, type: RemoveType):
    instance: MONAILabelApp = get_app_instance()
    if type == RemoveType.label:
        instance.datastore().remove_label(id)
    elif type == RemoveType.label_tag:
        instance.datastore().remove_label_by_tag(id)
    else:
        instance.datastore().remove_image(id)
    return {}


@router.put("/label", summary="Save Finished Label")
async def save_label(
    background_tasks: BackgroundTasks,
    image: str,
    tag: str = DefaultLabelTag.FINAL.value,
    label: UploadFile = File(...),
):
    file_ext = "".join(pathlib.Path(label.filename).suffixes) if label.filename else ".nii.gz"
    label_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
    tag = tag if tag else DefaultLabelTag.FINAL.value

    with open(label_file, "wb") as buffer:
        shutil.copyfileobj(label.file, buffer)
        background_tasks.add_task(remove_file, label_file)

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


@router.get("/image", summary="Download Image")
async def download_image(image):
    instance: MONAILabelApp = get_app_instance()
    image = instance.datastore().get_image_uri(image)
    if not os.path.isfile(image):
        raise HTTPException(status_code=404, detail="Image NOT Found")

    return FileResponse(image, media_type=get_mime_type(image), filename=os.path.basename(image))


@router.get("/label", summary="Download Label")
async def download_label(label):
    instance: MONAILabelApp = get_app_instance()
    label = instance.datastore().get_label_uri(label)
    if not os.path.isfile(label):
        raise HTTPException(status_code=404, detail="Label NOT Found")

    return FileResponse(label, media_type=get_mime_type(label), filename=os.path.basename(label))
