import logging
from typing import Optional

from fastapi import APIRouter

from monailabel.endpoints.utils import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["Train"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get Status of Training Task")
async def status(all: bool = False, check_if_running: bool = False):
    return BackgroundTask.status("train", all, check_if_running)


@router.post("/", summary="Run Training Task")
async def run(params: Optional[dict] = None):
    return BackgroundTask.run("train", params=params)


@router.delete("/", summary="Stop Training Task")
async def stop():
    return BackgroundTask.stop("train")
