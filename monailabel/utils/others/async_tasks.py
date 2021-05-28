import functools
import json
import logging
import os
import os.path
import platform
import subprocess
import uuid
from collections import deque
from datetime import datetime
from threading import Thread
from typing import Dict

from monailabel.config import settings

logger = logging.getLogger(__name__)

background_tasks: Dict = {"train": [], "infer": [], "batch_infer": []}
background_processes: Dict = {"train": dict(), "infer": dict(), "batch_infer": dict()}


def task_func(task, method):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    script = "run_monailabel_user_app.bat" if any(platform.win32_ver()) else "run_monailabel_user_app.sh"

    cmd = [
        os.path.realpath(os.path.join(base_dir, "scripts", script)),
        settings.APP_DIR,
        settings.STUDIES,
        method,
        json.dumps(task["request"]),
    ]

    process = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy(),
    )
    task_id = task["id"]
    background_processes[method][task_id] = process

    task["status"] = "RUNNING"
    task["details"] = deque(maxlen=20)

    plogger = logging.getLogger(f"task_{method}")
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            plogger.info(line)
            task["details"].append(line)

    logger.info("Return code: {}".format(process.returncode))
    background_processes[method].pop(task_id, None)
    process.stdout.close()

    task["end_ts"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    if task["status"] == "RUNNING":
        task["status"] = "DONE" if process.returncode == 0 else "ERROR"


def background_task(request, method):
    task = {
        "id": uuid.uuid4(),
        "status": "SUBMITTED",
        "request": request,
        "start_ts": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
    }

    background_tasks[method].append(task)
    thread = Thread(target=functools.partial(task_func, task, method))
    thread.start()
    return task


def stop_background_task(method):
    if not len(background_processes[method]):
        return None

    task_id, process = next(iter(background_processes[method].items()))
    process.kill()
    background_processes[method].pop(task_id, None)

    task = [task for task in background_tasks[method] if task["id"] == task_id][0]
    task["status"] = "STOPPED"
    task["end_ts"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return task


def tasks(method):
    return background_tasks[method]


def processes(method):
    return background_processes[method]
