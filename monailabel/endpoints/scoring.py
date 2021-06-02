import logging
from typing import Optional

from fastapi import APIRouter

from monailabel.endpoints.utils import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/scoring",
    tags=["Scoring"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", summary="Get Status of Scoring Task")
async def status(all: bool = False, check_if_running: bool = False):
    return BackgroundTask.status("scoring", all, check_if_running)


@router.post("/{method}", summary="Run Scoring Task")
async def run(method: str, params: Optional[dict] = None, force_sync: Optional[bool] = False):
    return BackgroundTask.run("scoring", request={"method": method}, params=params, force_sync=force_sync)


@router.delete("/", summary="Stop Scoring Task")
async def stop():
    return BackgroundTask.stop("scoring")
