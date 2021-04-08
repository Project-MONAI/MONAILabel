import logging

from fastapi import APIRouter, HTTPException
from starlette.background import BackgroundTasks

from server.interface import MONAIApp
from server.utils.app_utils import get_app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", summary="Run Training")
async def run_train(background_tasks: BackgroundTasks, epochs: int):
    request = {
        'device': "cuda",
        'epochs': epochs,
        'amp': True,
        'train': {},
        'val': {},
    }

    logger.info(f"Train Request: {request}")
    instance: MONAIApp = get_app_instance()
    result = instance.train(request)

    logger.info(f"Train Result: {result}")
    if result is None:
        raise HTTPException(status_code=500, detail=f"Failed to execute train")
    return result


@router.post("/stop", summary="Stop Training")
async def stop_train(background_tasks: BackgroundTasks):
    instance: MONAIApp = get_app_instance()
    return instance.stop_train({})
