import base64
import logging
import os
from typing import Dict, Optional

from fastapi import APIRouter

from monailabel.config import settings
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.generic import file_checksum, get_app_instance

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
    image_path = result["path"]
    image_id = result["id"]
    name = os.path.basename(image_path)

    digest = None
    if checksum:  # It's always costly operation (some clients to access directly from shared file-system)
        digest = cached_digest.get(image_path)
        digest = digest if digest is not None else file_checksum(image_path)
        digest = f"SHA256:{digest}"

    encoded = base64.urlsafe_b64encode(image_path.encode("utf-8")).decode("utf-8")
    encoded = encoded.rstrip("=")
    url = "/download/{}".format(encoded)

    return {
        "name": name,
        "id": image_id,
        "path": image_path,
        "studies": settings.STUDIES,
        "url": url,
        "checksum": digest,
    }
