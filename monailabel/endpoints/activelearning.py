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
from typing import Dict, Optional

from fastapi import APIRouter

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.app_utils import app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/activelearning",
    tags=["ActiveLearning"],
    responses={404: {"description": "Not found"}},
)

cached_digest: Dict = dict()


@router.post("/{strategy}", summary="Run Active Learning strategy to get next sample")
async def sample(strategy: str, params: Optional[dict] = None, checksum: Optional[bool] = True):
    request = {"strategy": strategy}

    instance: MONAILabelApp = app_instance()
    config = instance.info().get("config", {}).get("activelearning", {})
    request.update(config)

    params = params if params is not None else {}
    request.update(params)

    logger.info(f"Active Learning Request: {request}")
    result = instance.next_sample(request)
    if not result:
        return {}

    image_id = result["id"]
    image_info = instance.datastore().get_image_info(image_id)

    return {
        "id": image_id,
        **image_info,
    }
