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


def _task_func(task, method):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    script = "run_monailabel_app.bat" if any(platform.win32_ver()) else "run_monailabel_app.sh"
    if os.path.exists(os.path.realpath(os.path.join(base_dir, "scripts", script))):
        script = os.path.realpath(os.path.join(base_dir, "scripts", script))

    cmd = [
        script,
        settings.APP_DIR,
        settings.STUDIES,
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


def run_background_task(request, method, debug=False):
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
        thread = Thread(target=functools.partial(_task_func, task, method))
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


def run_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--app", required=True)
    parser.add_argument("-s", "--studies", required=True)
    parser.add_argument("-m", "--method", default="info")
    parser.add_argument("-r", "--request", default="{}")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()
    args.app = os.path.realpath(args.app)
    args.studies = os.path.realpath(args.studies)

    settings.APP_DIR = args.app
    settings.STUDIES = args.studies

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_background_task(json.loads(args.request), args.method, debug=True)


if __name__ == "__main__":
    run_main()
