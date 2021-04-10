import base64
import os
from typing import Optional

from fastapi import APIRouter

from monailabel.interface import MONAILabelApp
from monailabel.utils.app_utils import get_app_instance
from monailabel.utils.generic import file_checksum

router = APIRouter(
    prefix="/activelearning",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)

cached_digest = dict()


# TODO:: Return both name and binary image in the response
@router.post("/next_sample", summary="Run Active Learning strategy to get next sample")
async def next_sample(config: Optional[dict] = {"strategy": "random"}):
    request = config if config else dict()

    instance: MONAILabelApp = get_app_instance()
    result = instance.next_sample(request)
    image = result["image"]
    name = os.path.basename(image)

    checksum = cached_digest.get(image)
    checksum = checksum if checksum is not None else file_checksum(image)

    encoded = base64.urlsafe_b64encode(image.encode('utf-8')).decode('utf-8')
    encoded = encoded.rstrip('=')
    url = "/download/{}".format(encoded)

    return {
        "name": name,
        "image": image,
        "checksum": checksum,
        "url": url,
    }


@router.post("/save_label", summary="Save Finished Label")
async def save_label():
    instance: MONAILabelApp = get_app_instance()
    return instance.save_label({})
