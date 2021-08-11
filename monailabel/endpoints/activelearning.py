import logging
from typing import Dict, Optional

from fastapi import APIRouter

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.generic import get_app_instance

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

    instance: MONAILabelApp = get_app_instance()
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
