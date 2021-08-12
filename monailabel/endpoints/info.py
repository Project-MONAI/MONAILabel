from fastapi import APIRouter

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.others.app_utils import app_instance

router = APIRouter(
    prefix="/info",
    tags=["AppService"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get App Info")
async def app_info():
    instance: MONAILabelApp = app_instance()
    return instance.info()
