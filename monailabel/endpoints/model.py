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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from monailabel.endpoints.user.auth import User, get_basic_user
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import file_ext, get_mime_type

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


@router.get("/{model}", summary="Download Latest Model Weights")
async def api_download_model(model: str, user: User = Depends(get_basic_user)):
    return download_model(model)


@router.get("/info/{model}", summary="Get CheckSum/Details for the Latest Model File")
async def api_model_info(model: str, user: User = Depends(get_basic_user)):
    return model_info(model)
