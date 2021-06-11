import logging
from typing import Optional

import torch
from fastapi import APIRouter

from monailabel.endpoints.utils import BackgroundTask
from monailabel.interfaces.tasks.batch_infer import BatchInferImageType

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
async def run(
    model: str,
    images: Optional[BatchInferImageType] = BatchInferImageType.IMAGES_ALL,
    params: Optional[dict] = None,
    run_sync: Optional[bool] = False,
):
    request = {"model": model, "images": images}
    return BackgroundTask.run("batch_infer", request=request, params=params, force_sync=run_sync)


@router.delete("/infer", summary="Stop Batch Inference Task")
async def stop():
    res = BackgroundTask.stop("batch_infer")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res
