# Copyright 2020 - 2021 MONAI Consortium
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
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, Response
from requests_toolbelt import MultipartEncoder

from monailabel.datastore.utils.convert import binary_to_image
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.others.generic import get_mime_type, remove_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/infer",
    tags=["Infer"],
    responses={
        404: {"description": "Not found"},
        200: {
            "description": "OK",
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "points": {
                                "type": "string",
                                "description": "Reserved for future; Currently it will be empty",
                            },
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "The result NIFTI image which will have segmentation mask",
                            },
                        },
                    },
                    "encoding": {
                        "points": {"contentType": "text/plain"},
                        "file": {"contentType": "application/octet-stream"},
                    },
                },
                "application/json": {"schema": {"type": "string", "example": "{}"}},
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
)


class ResultType(str, Enum):
    image = "image"
    json = "json"
    all = "all"


def send_response(datastore, result, output, background_tasks):
    res_img = result.get("file") if result.get("file") else result.get("label")
    res_tag = result.get("tag")
    res_json = result.get("params")

    if res_img:
        if not os.path.exists(res_img):
            res_img = datastore.get_label_uri(res_img, res_tag)
        else:
            background_tasks.add_task(remove_file, res_img)

    if output == "json":
        return res_json

    m_type = get_mime_type(res_img)

    if output == "image":
        return FileResponse(res_img, media_type=m_type, filename=os.path.basename(res_img))

    res_fields = dict()
    res_fields["params"] = (None, json.dumps(res_json), "application/json")
    if res_img and os.path.exists(res_img):
        res_fields["image"] = (os.path.basename(res_img), open(res_img, "rb"), m_type)
    else:
        logger.info(f"Return only Result Json as Result Image is not available: {res_img}")
        return res_json

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)


def run_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str = "",
    session_id: str = "",
    params: str = Form("{}"),
    file: UploadFile = File(None),
    label: UploadFile = File(None),
    output: Optional[ResultType] = None,
):
    request = {"model": model, "image": image}

    if not file and not image and not session_id:
        raise HTTPException(status_code=500, detail="Neither Image nor File not Session ID input is provided")

    instance: MONAILabelApp = app_instance()

    if file:
        file_ext = "".join(pathlib.Path(file.filename).suffixes) if file.filename else ".nii.gz"
        image_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

        with open(image_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            request["image"] = image_file
            background_tasks.add_task(remove_file, image_file)

    if label:
        file_ext = "".join(pathlib.Path(label.filename).suffixes) if label.filename else ".nii.gz"
        label_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

        with open(label_file, "wb") as buffer:
            shutil.copyfileobj(label.file, buffer)
            background_tasks.add_task(remove_file, label_file)

        # if binary file received, e.g. scribbles from OHIF - then convert using reference image
        if file_ext == ".bin":
            image_uri = instance.datastore().get_image_uri(image)
            label_file = binary_to_image(image_uri, label_file)

        request["label"] = label_file

    config = instance.info().get("config", {}).get("infer", {})
    request.update(config)

    p = json.loads(params) if params else {}
    request.update(p)

    if session_id:
        session = instance.sessions().get_session(session_id)
        if session:
            request["image"] = session.image
            request["session"] = session.to_json()

    logger.info(f"Infer Request: {request}")
    result = instance.infer(request)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to execute infer")
    return send_response(instance.datastore(), result, output, background_tasks)


@router.post("/{model}", summary="Run Inference for supported model")
async def api_run_inference(
    background_tasks: BackgroundTasks,
    model: str,
    image: str = "",
    session_id: str = "",
    params: str = Form("{}"),
    file: UploadFile = File(None),
    label: UploadFile = File(None),
    output: Optional[ResultType] = None,
):
    return run_inference(background_tasks, model, image, session_id, params, file, label, output)
