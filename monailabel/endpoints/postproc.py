import json
import logging
import os
import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Optional

from fastapi import APIRouter
from fastapi import File, UploadFile
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from requests_toolbelt import MultipartEncoder
from starlette.background import BackgroundTasks

from monailabel.config import settings
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.app_utils import get_app_instance
from monailabel.utils.others.generic import file_checksum
from monailabel.utils.others.generic import get_mime_type

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/postproc",
    tags=["AppService"],
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
                                "description": "Reserved for future; Currently it will be empty"
                            },
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "The result NIFTI image which will have segmentation mask"
                            }
                        }
                    },
                    "encoding": {
                        "points": {
                            "contentType": "text/plain"
                        },
                        "file": {
                            "contentType": "application/octet-stream"
                        }
                    }
                },
                "application/json": {
                    "schema": {
                        "type": "string",
                        "example": "{}"
                    }
                },
                "application/octet-stream": {
                    "schema": {
                        "type": "string",
                        "format": "binary"
                    }
                }
            }
        }
    },
)

class ResultType(str, Enum):
    image = "image"
    json = "json"
    all = "all"

# cached_digest = dict()


def send_response(result, output, background_tasks):
    def remove_file(path: str) -> None:
        if os.path.exists(path):
            os.unlink(path)

    res_img = result.get('label')
    res_json = result.get('params')

    if output == 'json':
        return res_json

    background_tasks.add_task(remove_file, res_img)
    m_type = get_mime_type(res_img)

    if output == 'image':
        return FileResponse(res_img, media_type=m_type, filename=os.path.basename(res_img))

    res_fields = dict()
    res_fields['params'] = (None, json.dumps(res_json), 'application/json')
    res_fields['image'] = (os.path.basename(res_img), open(res_img, 'rb'), m_type)

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)


@router.post("/scrib", summary="Save Finished Label")
async def postproc_label(
    background_tasks: BackgroundTasks,
    method: str, 
    image: str, 
    scribbles: UploadFile = File(...),
    params: Optional[dict] = None,
    output: Optional[ResultType] = None):

    file_ext = ''.join(pathlib.Path(image).suffixes)
    scribbles_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

    with open(scribbles_file, "wb") as buffer:
        shutil.copyfileobj(scribbles.file, buffer)
    print('HERE')
    instance: MONAILabelApp = get_app_instance()
    request = {"method": method, "image": image, "scribbles": scribbles_file}

    params = params if params is not None else {}
    request.update(params)
    
    result = instance.postproc_label(request)

    if result is None:
        raise HTTPException(status_code=500, detail=f"Failed to execute infer")
    return send_response(result, output, background_tasks)
