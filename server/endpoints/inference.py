import logging
from enum import Enum
from typing import Optional

from fastapi import APIRouter
from starlette.background import BackgroundTasks

from server.utils.app_utils import get_app_instance, send_response

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


# TODO:: Run Inference for an item in dataset/session
@router.post("/{app}", summary="Run Infer action for an existing App")
async def run_inference(app: str, background_tasks: BackgroundTasks, output: Optional[ResultType] = None):
    request = {
        "image": "/workspace/Data/_image.nii.gz",
        "params": {}
    }

    instance, _ = get_app_instance(app, background_tasks)
    result = instance.infer(request=request)
    return send_response(app, result, output, background_tasks)
