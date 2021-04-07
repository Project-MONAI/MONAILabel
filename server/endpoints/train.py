import logging
import os

from fastapi import APIRouter, HTTPException
from starlette.background import BackgroundTasks

from server.core.config import settings
from server.internal.grpc.request import grpc_train
from server.utils.app_utils import app_info, get_grpc_port

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["AppEngine"],
    responses={404: {"description": "Not found"}},
)


@router.post("/{app}", summary="Run Train action for an existing App")
async def run_train(app: str, background_tasks: BackgroundTasks):
    info = app_info(app)
    app_dir = info['path']

    output_dir = os.path.join(app_dir, "model", "run_0")
    dataset_root = os.path.join(settings.WORKSPACE, "datasets", "Task09_Spleen")

    request = {
        'output_dir': output_dir,
        'data_list': os.path.join(dataset_root, "dataset.json"),
        'data_root': dataset_root,
        'device': "cuda",
        'epochs': 1,
        'amp': True,
        'train': {},
        'val': {},
    }

    logger.info(f"Train Request: {request}")
    result = await grpc_train(request, get_grpc_port(app))
    logger.info(f"Train Result: {result}")

    if result is None:
        raise HTTPException(status_code=500, detail=f"Failed to execute train for {app}")
    return result
