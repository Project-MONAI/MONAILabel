import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from monailabel.interfaces.tasks import processes, tasks, background_task, stop_background_task
from monailabel.utils.others.generic import get_app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", summary="Run Training Task")
async def run_train(params: Optional[dict] = None):
    train_process = processes('train')
    if len(train_process):
        raise HTTPException(status_code=429, detail=f"Training is Already Running")

    request = {}
    config = get_app_instance().info().get("config", {}).get("train", {})
    request.update(config)

    params = params if params is not None else {}
    request.update(params)

    logger.info(f"Train Request: {request}")
    return background_task(request, 'train')


@router.get("/", summary="Get Status of Training Task")
async def status(all: bool = False, check_if_running: bool = False):
    train_process = processes('train')
    train_tasks = tasks('train')

    if check_if_running:
        if len(train_process) == 0:
            raise HTTPException(status_code=404, detail=f"No Training Tasks are currently Running")
        task_id = next(iter(train_process))
        return [task for task in train_tasks if task["id"] == task_id][0]

    task = train_tasks[-1] if len(train_tasks) else None
    if task is None:
        raise HTTPException(status_code=404, detail=f"No Training Tasks Found")

    return train_tasks if all else task


@router.delete("/", summary="Stop Training Task")
async def stop_train():
    return stop_background_task('train')
