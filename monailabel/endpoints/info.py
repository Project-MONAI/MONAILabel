from fastapi import APIRouter

from monailabel.interface import MONAILabelApp
from monailabel.utils.others.app_utils import get_app_instance

router = APIRouter(
    prefix="/info",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get App Info")
async def app_info():
    instance: MONAILabelApp = get_app_instance()
    return instance.info()
