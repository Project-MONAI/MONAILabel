from fastapi import APIRouter

from server.interface import MONAIApp
from server.utils.app_utils import get_app_instance

router = APIRouter(
    prefix="/info",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get App Info")
async def app_info():
    instance: MONAIApp = get_app_instance()
    return instance.info()
