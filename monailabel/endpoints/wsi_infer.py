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
from typing import Optional, Sequence, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from monailabel.endpoints.user.auth import User, get_basic_user
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import get_mime_type, remove_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/infer",
    tags=["Infer"],
    responses={404: {"description": "Not found"}},
)


class WSIInput(BaseModel):
    level: Optional[int] = Field(0, title="Resolution Level")
    location: Optional[Sequence[int]] = Field([0, 0], title="Location of Region")
    size: Optional[Sequence[int]] = Field([2048, 2048], title="Size of Region")
    tile_size: Optional[Sequence[int]] = Field([1024, 1024], title="Tile size")
    min_poly_area: Optional[int] = Field(80, title="Min Area to filter mask polygons")
    params: Optional[dict] = Field({}, title="Additional Params")


class ResultType(str, Enum):
    asap = "asap"
    dsa = "dsa"
    json = "json"


def send_response(datastore, result, output, background_tasks):
    res_img = result.get("file") if result.get("file") else result.get("label")
    res_json = result.get("params")

    if res_img:
        if not os.path.exists(res_img):
            res_img = datastore.get_label_uri(res_img, result.get("tag"))
        else:
            background_tasks.add_task(remove_file, res_img)

    if not res_img or output == "json":
        return res_json

    m_type = get_mime_type(res_img)
    return FileResponse(res_img, media_type=m_type, filename=os.path.basename(res_img))


def run_wsi_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str = "",
    session_id: str = "",
    file: Union[UploadFile, None] = None,
    wsi: WSIInput = WSIInput(),
    output: Optional[ResultType] = ResultType.dsa,
):
    request = {"model": model, "image": image, "output": output.value if output else None}

    if not file and not image and not session_id:
        raise HTTPException(status_code=500, detail="Neither Image nor File not Session ID input is provided")

    instance: MONAILabelApp = app_instance()

    if file and file.filename:
        file_ext = "".join(pathlib.Path(file.filename).suffixes) if file.filename else ".png"
        image_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

        with open(image_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            request["image"] = image_file
            background_tasks.add_task(remove_file, image_file)

    config = instance.info().get("config", {}).get("infer", {})
    request.update(config)
    request.update(wsi.dict(exclude={"params"}))
    if wsi.params:
        request.update(wsi.params)

    if session_id:
        session = instance.sessions().get_session(session_id)
        if session:
            request["image"] = session.image
            request["session"] = session.to_json()

    logger.info(f"WSI Infer Request: {request}")

    result = instance.infer_wsi(request)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to execute wsi infer")
    return send_response(instance.datastore(), result, output, background_tasks)


@router.post("/wsi/{model}", summary="Run WSI Inference for supported model", deprecated=True)
async def api_run_wsi_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str = "",
    session_id: str = "",
    wsi: WSIInput = WSIInput(),
    output: Optional[ResultType] = None,
    user: User = Depends(get_basic_user),
):
    return run_wsi_inference(background_tasks, model, image, session_id, None, wsi, output)


@router.post("/wsi_v2/{model}", summary="Run WSI Inference for supported model")
async def api_run_wsi_v2_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str = "",
    session_id: str = "",
    file: UploadFile = File(None),
    wsi: str = Form(WSIInput().json()),
    output: Optional[ResultType] = None,
    user: User = Depends(get_basic_user),
):
    w = WSIInput.parse_obj(json.loads(wsi))
    return run_wsi_inference(background_tasks, model, image, session_id, file, w, output)
