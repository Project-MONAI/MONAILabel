from fastapi import APIRouter

from server.interface import MONAIApp
from server.utils.app_utils import get_app_instance

router = APIRouter(
    prefix="/activelearning",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


# TODO:: Return both name and binary image in the response
@router.post("/next_sample", summary="Run Active Learning strategy to get next sample")
async def next_sample():
    instance: MONAIApp = get_app_instance()
    return instance.next_sample({})


@router.post("/save_label", summary="Save Finished Label")
async def save_label():
    instance: MONAIApp = get_app_instance()
    return instance.save_label({})
