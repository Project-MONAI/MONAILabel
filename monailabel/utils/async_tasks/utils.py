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
import json
import logging
import os
import os.path
import random
import subprocess
import sys
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict

import psutil

logger = logging.getLogger(__name__)

background_tasks: Dict = {}
background_processes: Dict = {}
background_executors: Dict = {}


def _task_func(task, method, callback=None):
    request = task["request"]
    my_env = {**os.environ}

    gpus = request.get("gpus", "all")
    gpus = gpus if gpus else "all"
    if gpus != "all":
        my_env["CUDA_VISIBLE_DEVICES"] = gpus
    request["gpus"] = "all"

    if method == "train":
        my_env["MONAI_LABEL_DATASTORE_AUTO_RELOAD"] = "false"
        my_env["MASTER_ADDR"] = "127.0.0.1"
        my_env["MASTER_PORT"] = str(random.randint(1234, 1334))

    cmd = [
        sys.executable,
        "-m",
        "monailabel.interfaces.utils.app",
        "-m",
        method,
        "-r",
        json.dumps(request, separators=(",", ":")),
    ]

    logger.info(f"COMMAND:: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, universal_newlines=True, env=my_env
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
        "id": str(uuid.uuid4()),
        "status": "SUBMITTED",
        "request": request,
        "start_ts": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if background_tasks.get(method) is None:
        background_tasks[method] = []
    if background_processes.get(method) is None:
        background_processes[method] = dict()
    if background_executors.get(method) is None:
        background_executors[method] = ThreadPoolExecutor(max_workers=1)

    background_tasks[method].append(task)
    if debug:
        _task_func(task, method)
    else:
        executor = background_executors[method]
        executor.submit(_task_func, task, method, callback)
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
