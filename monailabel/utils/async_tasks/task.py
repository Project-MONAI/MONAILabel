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

import logging

from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.async_tasks.utils import processes, run_background_task, stop_background_task, tasks

logger = logging.getLogger(__name__)


class AsyncTask:
    @staticmethod
    def run(method: str, request=None, params=None, force_sync=False, enqueue=False):
        if len(processes(method)) and not enqueue:
            description = f"++++++++++ {method.capitalize()} Task is Already Running"
            logger.info(description)
            return None, description

        instance = app_instance()
        config = instance.info().get("config", {}).get(method, {})
        request = request if request else {}
        request.update(config)

        params = params if params is not None else {}
        request.update(params)

        logger.info(f"{method.capitalize()} request: {request}")
        if force_sync:
            if method == "batch_infer":
                return instance.batch_infer(request), None
            if method == "scoring":
                return instance.scoring(request), None
            if method == "train":
                return instance.train(request), None

        return run_background_task(request, method), None

    @staticmethod
    def status(method: str, all: bool = False, check_if_running: bool = False):
        batch_process = processes(method)
        batch_tasks = tasks(method)

        if check_if_running:
            if len(batch_process) == 0:
                description = f"No {method.capitalize()} Tasks are currently Running"
                logger.debug(description)
                return None, description
            task_id = next(iter(batch_process))
            return [task for task in batch_tasks if task["id"] == task_id][0], None

        task = batch_tasks[-1] if len(batch_tasks) else None
        if task is None:
            description = f"No {method.capitalize()} Tasks Found"
            logger.debug(description)
            return None, description

        ret = batch_tasks if all else task
        return ret, None

    @staticmethod
    def stop(method):
        return stop_background_task(method)
