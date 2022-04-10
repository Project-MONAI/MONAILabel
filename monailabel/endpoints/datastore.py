# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.datastore import Datastore, DefaultLabelTag
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import get_mime_type, remove_file

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


def datastore(output: Optional[ResultType] = None):
    d: Datastore = app_instance().datastore()
    output = output if output else ResultType.stats

    logger.debug(f"output type: {output}")
    if output == ResultType.all:
        return d.json()
    if output == ResultType.train:
        return d.datalist()
    return d.status()


def add_image(
    background_tasks: BackgroundTasks,
    image: Optional[str] = None,
    params: str = Form("{}"),
    file: UploadFile = File(...),
):
    logger.info(f"Image: {image}; File: {file}; params: {params}")
    file_ext = "".join(pathlib.Path(file.filename).suffixes) if file.filename else ".nii.gz"

    image_id = image if image else os.path.basename(file.filename).replace(file_ext, "")
    image_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

    with open(image_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(remove_file, image_file)

    instance: MONAILabelApp = app_instance()
    save_params: Dict[str, Any] = json.loads(params) if params else {}
    image_id = instance.datastore().add_image(image_id, image_file, save_params)
    return {"image": image_id}


def remove_image(id: str):
    instance: MONAILabelApp = app_instance()
    instance.datastore().remove_image(id)
    return {}


def save_label(
    background_tasks: BackgroundTasks,
    image: str,
    params: str = Form("{}"),
    tag: str = DefaultLabelTag.FINAL.value,
    label: UploadFile = File(...),
):
    file_ext = "".join(pathlib.Path(label.filename).suffixes) if label.filename else ".nii.gz"
    label_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
    tag = tag if tag else DefaultLabelTag.FINAL.value

    with open(label_file, "wb") as buffer:
        shutil.copyfileobj(label.file, buffer)
        background_tasks.add_task(remove_file, label_file)

    instance: MONAILabelApp = app_instance()
    save_params: Dict[str, Any] = json.loads(params) if params else {}
    logger.info(f"Save Label params: {params}")

    label_id = instance.datastore().save_label(image, label_file, tag, save_params)
    res = instance.on_save_label(image, label_id)
    res = res if res else {}
    res.update(
        {
            "image": image,
            "label": label_id,
        }
    )
    return res


def remove_label(id: str, tag: str):
    instance: MONAILabelApp = app_instance()
    instance.datastore().remove_label(id, tag)
    return {}


def download_image(image: str):
    instance: MONAILabelApp = app_instance()
    image = instance.datastore().get_image_uri(image)
    if not os.path.isfile(image):
        raise HTTPException(status_code=404, detail="Image NOT Found")

    return FileResponse(image, media_type=get_mime_type(image), filename=os.path.basename(image))


def download_label(label: str, tag: str):
    instance: MONAILabelApp = app_instance()
    label = instance.datastore().get_label_uri(label, tag)
    if not os.path.isfile(label):
        raise HTTPException(status_code=404, detail="Label NOT Found")

    return FileResponse(label, media_type=get_mime_type(label), filename=os.path.basename(label))


@router.get("/", summary="Get All Images/Labels from datastore")
async def api_datastore(output: Optional[ResultType] = None):
    return datastore(output)


@router.put("/", summary="Upload new Image")
async def api_add_image(
    background_tasks: BackgroundTasks,
    image: Optional[str] = None,
    params: str = Form("{}"),
    file: UploadFile = File(...),
):
    return add_image(background_tasks, image, params, file)


@router.delete("/", summary="Remove Image and corresponding labels")
async def api_remove_image(id: str):
    return remove_image(id)


@router.put("/label", summary="Save Finished Label")
async def api_save_label(
    background_tasks: BackgroundTasks,
    image: str,
    params: str = Form("{}"),
    tag: str = DefaultLabelTag.FINAL.value,
    label: UploadFile = File(...),
):
    return save_label(background_tasks, image, params, tag, label)


@router.delete("/label", summary="Remove Label")
async def api_remove_label(id: str, tag: str):
    return remove_label(id, tag)


@router.get("/image", summary="Download Image")
async def api_download_image(image: str):
    return download_image(image)


@router.get("/label", summary="Download Label")
async def api_download_label(label: str, tag: str):
    return download_label(label, tag)
