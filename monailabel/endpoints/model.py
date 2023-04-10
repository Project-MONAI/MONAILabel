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

import logging
import os
import pathlib
import shutil
import tempfile
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse

from monailabel.config import RBAC_ADMIN, settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import file_checksum, file_ext, get_mime_type, remove_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/model",
    tags=["Model"],
    responses={404: {"description": "Not found"}},
)


def model_info(model: str):
    instance: MONAILabelApp = app_instance()
    info = instance.model_info(model)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model File NOT Found for {model}")
    return info


def download_model(model: str):
    logger.info(f"Download model file for: {model}")

    instance: MONAILabelApp = app_instance()
    file = instance.model_file(model)
    if not file or not os.path.exists(file):
        raise HTTPException(status_code=404, detail=f"Model File NOT Found for {model}")

    filename = f"{model}{file_ext(file)}"
    return FileResponse(file, media_type=get_mime_type(file), filename=filename)


def update_model(background_tasks: BackgroundTasks, model: str, file: UploadFile):
    ext = "".join(pathlib.Path(file.filename).suffixes) if file.filename else ".pt"
    model_file = tempfile.NamedTemporaryFile(suffix=ext).name

    with open(model_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(remove_file, model_file)

    instance: MONAILabelApp = app_instance()
    prev_file = instance.model_file(model, validate=False)
    if not prev_file:
        raise HTTPException(status_code=500, detail=f"Model File Name NOT configured for {model}")

    if not os.path.exists(prev_file):
        logger.info(f"Previous Model File [{prev_file}] NOT Found for {model}; Adding new one!")

    logger.info(f"Updating Model File for model: {model}; {model_file} => {prev_file}")
    shutil.copy(model_file, prev_file)

    s = os.stat(prev_file)
    checksum = file_checksum(prev_file)
    return {"checksum": checksum, "modified_time": int(s.st_mtime)}


def delete_model(model: str):
    logger.info(f"Delete model file for: {model}")

    instance: MONAILabelApp = app_instance()
    file = instance.model_file(model)
    if not file or not os.path.exists(file):
        raise HTTPException(status_code=404, detail=f"Model File NOT Found for {model}")

    shutil.move(file, f"{file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.deleted")
    return {}


@router.get("/{model}", summary=f"{RBAC_ADMIN}Download Latest Model Weights")
async def api_download_model(
    model: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ADMIN)),
):
    return download_model(model)


@router.get("/info/{model}", summary=f"{RBAC_ADMIN}Get CheckSum/Details for the Latest Model File")
async def api_model_info(
    model: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ADMIN)),
):
    return model_info(model)


@router.put("/{model}", summary=f"{RBAC_ADMIN}Upload/Update Model File")
async def api_update_model(
    background_tasks: BackgroundTasks,
    model: str,
    file: UploadFile,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR)),
):
    return update_model(background_tasks, model, file)


@router.delete("/{model}", summary=f"{RBAC_ADMIN}Delete Model File")
async def api_delete_model(
    model: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR)),
):
    return delete_model(model)
