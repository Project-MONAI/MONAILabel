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
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.async_tasks.task import AsyncTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["Train"],
    responses={404: {"description": "Not found"}},
)


def status(all: bool = False, check_if_running: bool = False):
    res, detail = AsyncTask.status("train", all, check_if_running)
    if res is None:
        raise HTTPException(status_code=404, detail=detail)
    return res


def run(params: Optional[dict] = None, run_sync: Optional[bool] = False):
    instance: MONAILabelApp = app_instance()
    result = {}
    for model in instance.info()["trainers"]:
        request = {"model": model}
        if params and params.get(model):
            request.update(params[model])
        res, detail = AsyncTask.run("train", request=request, params=params, force_sync=run_sync, enqueue=True)
        result[model] = {"result": res, "detail": detail}
    return result


def run_model(
    model: str, params: Optional[dict] = None, run_sync: Optional[bool] = False, enqueue: Optional[bool] = False
):
    request = {"model": model} if model else {}
    res, detail = AsyncTask.run("train", request=request, params=params, force_sync=run_sync, enqueue=enqueue)
    if res is None:
        raise HTTPException(status_code=429, detail=detail)
    return res


def stop():
    res = AsyncTask.stop("train")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


@router.get("/", summary="Get Status of Training Task")
async def api_status(all: bool = False, check_if_running: bool = False):
    return status(all, check_if_running)


@router.post("/", summary="Run All Training Tasks")
async def api_run(params: Optional[dict] = None, run_sync: Optional[bool] = False):
    return run(params, run_sync)


@router.post("/{model}", summary="Run Training Task for specific model")
async def api_run_model(
    model: str, params: Optional[dict] = None, run_sync: Optional[bool] = False, enqueue: Optional[bool] = False
):
    return run_model(model, params, run_sync, enqueue)


@router.delete("/", summary="Stop Training Task")
async def api_stop():
    return stop()
