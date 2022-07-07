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
from monailabel.interfaces.tasks.batch_infer import BatchInferImageType
from monailabel.utils.async_tasks.task import AsyncTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/batch",
    tags=["Infer"],
    responses={404: {"description": "Not found"}},
)


def status(all: bool = False, check_if_running: bool = False):
    res, detail = AsyncTask.status("batch_infer", all, check_if_running)
    if res is None:
        raise HTTPException(status_code=404, detail=detail)
    return res


def run(
    model: str,
    images: Optional[BatchInferImageType] = BatchInferImageType.IMAGES_ALL,
    params: Optional[dict] = None,
    run_sync: Optional[bool] = False,
):
    request = {"model": model, "images": images}
    res, detail = AsyncTask.run("batch_infer", request=request, params=params, force_sync=run_sync)
    if res is None:
        raise HTTPException(status_code=429, detail=detail)
    return res


def stop():
    res = AsyncTask.stop("batch_infer")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


@router.get("/infer", summary="Get Status of Batch Inference Task")
async def api_status(all: bool = False, check_if_running: bool = False, user: User = Depends(get_annotator_user)):
    return status(all, check_if_running)


@router.post("/infer/{model}", summary="Run Batch Inference Task")
async def api_run(
    model: str,
    images: Optional[BatchInferImageType] = BatchInferImageType.IMAGES_ALL,
    params: Optional[dict] = None,
    run_sync: Optional[bool] = False,
    user: User = Depends(get_annotator_user),
):
    return run(model, images, params, run_sync)


@router.delete("/infer", summary="Stop Batch Inference Task")
async def api_stop(user: User = Depends(get_annotator_user)):
    return stop()
