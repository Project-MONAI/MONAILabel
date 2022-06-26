# Copyright (c) MONAI Consortium
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
from fastapi import APIRouter, Depends, HTTPException

from monailabel.endpoints.user.auth import User, get_annotator_user
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.utils.async_tasks.task import AsyncTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/scoring",
    tags=["Scoring"],
    responses={404: {"description": "Not found"}},
)


def status(all: bool = False, check_if_running: bool = False):
    res, detail = AsyncTask.status("scoring", all, check_if_running)
    if res is None:
        raise HTTPException(status_code=404, detail=detail)
    return res


def run(params: Optional[dict] = None, run_sync: Optional[bool] = False):
    instance: MONAILabelApp = app_instance()
    result = {}
    for method in instance.info()["scoring"]:
        request = {"method": method}
        if params and params.get(method):
            request.update(params[method])
        res, detail = AsyncTask.run("scoring", request=request, params=params, force_sync=run_sync, enqueue=True)
        result[method] = {"result": res, "detail": detail}
    return result


def run_method(method: str, params: Optional[dict] = None, run_sync: Optional[bool] = False):
    res, detail = AsyncTask.run("scoring", request={"method": method}, params=params, force_sync=run_sync)
    if res is None:
        raise HTTPException(status_code=429, detail=detail)
    return res


def stop():
    res = AsyncTask.stop("scoring")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


@router.get("/", summary="Get Status of Scoring Task")
async def api_status(
    all: bool = False,
    check_if_running: bool = False,
    user: User = Depends(get_annotator_user),
):
    return status(all, check_if_running)


@router.post("/", summary="Run All Scoring Tasks")
async def api_run(
    params: Optional[dict] = None,
    run_sync: Optional[bool] = False,
    user: User = Depends(get_annotator_user),
):
    return run(params, run_sync)


@router.post("/{method}", summary="Run Scoring Task for specific method")
async def api_run_method(
    method: str,
    params: Optional[dict] = None,
    run_sync: Optional[bool] = False,
    user: User = Depends(get_annotator_user),
):
    return run_method(method, params, run_sync)


@router.delete("/", summary="Stop Scoring Task")
async def api_stop(user: User = Depends(get_annotator_user)):
    return stop()
