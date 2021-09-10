# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import psutil

from monailabel.config import settings

logger = logging.getLogger(__name__)

background_tasks: Dict = {}
background_processes: Dict = {}


def _task_func(task, method, callback=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    script = "run_monailabel_app.bat" if any(platform.win32_ver()) else "run_monailabel_app.sh"
    if os.path.exists(os.path.realpath(os.path.join(base_dir, "scripts", script))):
        script = os.path.realpath(os.path.join(base_dir, "scripts", script))

    cmd = [
        script,
        settings.MONAI_LABEL_APP_DIR,
        settings.MONAI_LABEL_STUDIES,
        method,
        json.dumps(task["request"]),
    ]

    logger.info(f"COMMAND:: {' '.join(cmd)}")
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

    if callback:
        callback(task)


def run_background_task(request, method, callback=None, debug=False):
    task = {
        "id": uuid.uuid4(),
        "status": "SUBMITTED",
        "request": request,
        "start_ts": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if background_tasks.get(method) is None:
        background_tasks[method] = []
    if background_processes.get(method) is None:
        background_processes[method] = dict()

    background_tasks[method].append(task)
    if debug:
        _task_func(task, method)
    else:
        thread = Thread(target=functools.partial(_task_func, task, method, callback))
        thread.start()
    return task


def stop_background_task(method):
    logger.info(f"Kill background task for {method}")
    if not background_tasks.get(method) or not background_processes.get(method):
        return None

    task_id, process = next(iter(background_processes[method].items()))
    children = psutil.Process(pid=process.pid).children(recursive=True)
    for child in children:
        logger.info(f"Kill:: Child pid is {child.pid}")
        child.kill()
    logger.info(f"Kill:: Process pid is {process.pid}")
    process.kill()

    background_processes[method].pop(task_id, None)
    logger.info(f"Killed background process: {process.pid}")

    task = [task for task in background_tasks[method] if task["id"] == task_id][0]
    task["status"] = "STOPPED"
    task["end_ts"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return task


def tasks(method):
    """
    Returns List of all task ids
    """
    return background_tasks.get(method, [])


def processes(method):
    """
    Returns Dict of all task id => process
    """
    return background_processes.get(method, dict())
