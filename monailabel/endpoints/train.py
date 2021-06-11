import logging
from typing import Optional

import torch
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
async def run(params: Optional[dict] = None, run_sync: Optional[bool] = False):
    return BackgroundTask.run("train", params=params, force_sync=run_sync)


@router.delete("/", summary="Stop Training Task")
async def stop():
    res = BackgroundTask.stop("train")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res
