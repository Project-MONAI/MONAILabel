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

from fastapi import HTTPException

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.app_utils import app_instance
from monailabel.utils.others.async_tasks import processes, run_background_task, stop_background_task, tasks

logger = logging.getLogger(__name__)


class BackgroundTask:
    @staticmethod
    def run(method: str, request=None, params=None, force_sync=False):
        if len(processes(method)):
            raise HTTPException(status_code=429, detail=f"{method.capitalize()} Task is Already Running")

        instance: MONAILabelApp = app_instance()
        config = instance.info().get("config", {}).get(method, {})
        request = request if request else {}
        request.update(config)

        params = params if params is not None else {}
        request.update(params)

        logger.info(f"{method.capitalize()} request: {request}")
        if force_sync:
            if method == "batch_infer":
                return instance.batch_infer(request)
            if method == "scoring":
                return instance.scoring(request)
            if method == "train":
                return instance.train(request)

        return run_background_task(request, method)

    @staticmethod
    def status(method: str, all: bool = False, check_if_running: bool = False):
        batch_process = processes(method)
        batch_tasks = tasks(method)

        if check_if_running:
            if len(batch_process) == 0:
                raise HTTPException(status_code=404, detail=f"No {method.capitalize()} Tasks are currently Running")
            task_id = next(iter(batch_process))
            return [task for task in batch_tasks if task["id"] == task_id][0]

        task = batch_tasks[-1] if len(batch_tasks) else None
        if task is None:
            raise HTTPException(status_code=404, detail=f"No {method.capitalize()} Tasks Found")

        return batch_tasks if all else task

    @staticmethod
    def stop(method):
        return stop_background_task(method)
