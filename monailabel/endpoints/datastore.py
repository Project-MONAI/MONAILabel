import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from monailabel.interfaces import Datastore
from monailabel.utils.others.generic import get_app_instance

logger = logging.getLogger(__name__)
train_tasks = []
train_process = dict()

router = APIRouter(
    prefix="/datastore",
    tags=["Datastore"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get All Images/Labels from datastore")
async def datastore():
    d: Datastore = get_app_instance().datastore()
    return d.datalist()


@router.put("/", summary="Upload new Image")
async def add_image(image: str, file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail=f"Not Implemented Yet")


@router.delete("/", summary="Remove Image/Label")
async def remove_image(image: str):
    raise HTTPException(status_code=501, detail=f"Not Implemented Yet")
