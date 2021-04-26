import asyncio
import functools
import json
import logging
import os.path
import subprocess
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi import BackgroundTasks

from monailabel.config import settings
from monailabel.utils.app_utils import get_app_instance

logger = logging.getLogger(__name__)
train_tasks = []
train_process = dict()

router = APIRouter(
    prefix="/train",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


def train_func(task):
    import monailabel.utils.app_utils as app_utils
    cmd = [
        "python3",
        os.path.realpath(app_utils.__file__),
        "-a",
        settings.APP_DIR,
        "-s",
        settings.STUDIES,
        "train",
        "-r",
        json.dumps(task["request"])
    ]

    process = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy()
    )
    task_id = task["id"]
    train_process[task_id] = process

    task["status"] = "RUNNING"
    task["details"] = deque(maxlen=20)
    plogger = logging.getLogger("training")
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            plogger.info(line)
            task["details"].append(line)

    logger.info('Return code: {}'.format(process.returncode))
    train_process.pop(task_id, None)
    process.stdout.close()

    task["end_ts"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    if task["status"] == "RUNNING":
        task["status"] = "DONE" if process.returncode == 0 else "ERROR"


async def train_background_task(task) -> None:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, functools.partial(train_func, task))


@router.post("/", summary="Run Training Task")
async def run_train(background_tasks: BackgroundTasks, params: Optional[dict] = None):
    if len(train_process):
        raise HTTPException(status_code=429, detail=f"Training is Already Running")

    request = {}
    config = get_app_instance().info().get("config", {}).get("train", {})
    request.update(config)

    params = params if params is not None else {}
    request.update(params)

    logger.info(f"Train Request: {request}")
    task = {
        "id": uuid.uuid4(),
        "status": "SUBMITTED",
        "request": request,
        "start_ts": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
    }

    train_tasks.append(task)
    background_tasks.add_task(train_background_task, task)
    return task


@router.get("/", summary="Get Status of Training Task")
async def status(all: bool = False, check_if_running: bool = False):
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
    if len(train_process) == 0:
        raise HTTPException(status_code=404, detail=f"No Training Tasks are currently Running")

    task_id = next(iter(train_process))
    process = train_process[task_id]
    process.kill()
    train_process.pop(task_id, None)

    task = [task for task in train_tasks if task["id"] == task_id][0]
    task["status"] = "STOPPED"
    task["end_ts"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return task
