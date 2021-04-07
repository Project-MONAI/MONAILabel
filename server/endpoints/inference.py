import json
import logging
import mimetypes
import os
from enum import Enum
from typing import Optional

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from requests_toolbelt import MultipartEncoder
from starlette.background import BackgroundTasks

from server.internal.grpc.request import grpc_inference
from server.utils.app_utils import get_grpc_port

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inference",
    tags=["AppEngine"],
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


def send_response(result, output, background_tasks):
    def remove_file(path: str) -> None:
        os.unlink(path)

    res_img = result.get('label')
    res_json = result.get('params')

    if res_img is None or output == 'json':
        return res_json

    background_tasks.add_task(remove_file, res_img)
    m_type = mimetypes.guess_type(res_img, strict=False)
    logger.debug(f"Guessed Mime Type for Image: {m_type}")

    if m_type is None or m_type[0] is None:
        m_type = "application/octet-stream"
    else:
        m_type = f"{m_type[0]}/{m_type[1]}"
    logger.debug(f"Final Mime Type: {m_type}")

    if res_json is None or not len(res_json) or output == 'image':
        return FileResponse(res_img, media_type=m_type)

    res_fields = dict()
    res_fields['params'] = (None, json.dumps(res_json), 'application/json')
    res_fields['image'] = (os.path.basename(res_img), open(res_img, 'rb'), m_type)

    return_message = MultipartEncoder(fields=res_fields)
    return Response(content=return_message.to_string(), media_type=return_message.content_type)


# TODO:: Run Inference for an item in dataset/session
@router.post("/{app}", summary="Run Infer action for an existing App")
async def run_inference(app: str, background_tasks: BackgroundTasks, output: Optional[ResultType] = None):
    request = {
        "image": "/workspace/Data/_image.nii.gz",
        "params": {}
    }

    logger.info(f"Infer Request: {request}")
    result = await grpc_inference(request, get_grpc_port(app))
    logger.info(f"Infer Result: {result}")

    if result is None:
        raise HTTPException(status_code=500, detail=f"Failed to execute infer for {app}")
    return send_response(result, output, background_tasks)
