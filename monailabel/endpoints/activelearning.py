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
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends

from monailabel.config import RBAC_USER, settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/activelearning",
    tags=["ActiveLearning"],
    responses={404: {"description": "Not found"}},
)

cached_digest: Dict = dict()


def sample(strategy: str, params: Optional[dict] = None, user: Optional[str] = None):
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

    strategy_info = image_info.get("strategy", {})
    strategy_info[strategy] = {"ts": int(time.time()), "client_id": params.get("client_id", user)}
    try:
        instance.datastore().update_image_info(image_id, {"strategy": strategy_info})
    except:
        logger.warning(f"Failed to update Image info for {image_id}")

    result.update(image_info)
    logger.info(f"Next sample: {result}")
    return result


@router.post("/{strategy}", summary=f"{RBAC_USER}Run Active Learning strategy to get next sample")
async def api_sample(
    strategy: str,
    params: Optional[dict] = None,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    return sample(strategy, params, user.username)
