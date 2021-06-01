import logging
from typing import Optional

from fastapi import APIRouter

from monailabel.endpoints.utils import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/batch",
    tags=["Infer"],
    responses={404: {"description": "Not found"}},
)


@router.get("/infer", summary="Get Status of Batch Inference Task")
async def status(all: bool = False, check_if_running: bool = False):
    return BackgroundTask.status("batch_infer", all, check_if_running)


@router.post("/infer/{model}", summary="Run Batch Inference Task")
async def run(model: str, params: Optional[dict] = None, force_sync: Optional[bool] = True):
    return BackgroundTask.run("batch_infer", request={"model": model}, params=params, force_sync=force_sync)


@router.delete("/infer", summary="Stop Batch Inference Task")
async def stop():
    return BackgroundTask.stop("batch_infer")
