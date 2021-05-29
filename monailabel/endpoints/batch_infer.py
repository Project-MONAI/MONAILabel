import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from monailabel.utils.others.async_tasks import processes, run_background_task, stop_background_task, tasks
from monailabel.utils.others.generic import get_app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/batch",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.post("/infer/{model}", summary="Run Batch Inference Task")
async def run_infer(model: str, params: Optional[dict] = None):
    batch_process = processes("batch_infer")
    if len(batch_process):
        raise HTTPException(status_code=429, detail="Batch Inference is Already Running")

    request = {"model": model}
    config = get_app_instance().info().get("config", {}).get("batch_infer", {})
    request.update(config)

    params = params if params is not None else {}
    request.update(params)

    logger.info(f"Batch Infer Request: {request}")
    return run_background_task(request, "batch_infer")


@router.get("/infer", summary="Get Status of Batch Inference Task")
async def status(all: bool = False, check_if_running: bool = False):
    batch_process = processes("batch_infer")
    batch_tasks = tasks("batch_infer")

    if check_if_running:
        if len(batch_process) == 0:
            raise HTTPException(status_code=404, detail="No Batch Inference Tasks are currently Running")
        task_id = next(iter(batch_process))
        return [task for task in batch_tasks if task["id"] == task_id][0]

    task = batch_tasks[-1] if len(batch_tasks) else None
    if task is None:
        raise HTTPException(status_code=404, detail="No Batch Inference Tasks Found")

    return batch_tasks if all else task


@router.delete("/infer", summary="Stop Batch Inference Task")
async def stop_infer():
    return stop_background_task("batch_infer")
