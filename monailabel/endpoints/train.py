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
from fastapi import APIRouter

from monailabel.endpoints.utils import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/train",
    tags=["Train"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get Status of Training Task")
async def status(all: bool = False, check_if_running: bool = False):
    return BackgroundTask.status("train", all, check_if_running)


@router.post("/", summary="Run Training Task")
async def run(params: Optional[dict] = None, run_sync: Optional[bool] = False):
    return BackgroundTask.run("train", params=params, force_sync=run_sync)


@router.delete("/", summary="Stop Training Task")
async def stop():
    res = BackgroundTask.stop("train")

    # Try to clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res
