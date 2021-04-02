import logging

from fastapi import APIRouter, HTTPException

from server.utils.scanning import scan_apps

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/app",
    tags=["App"],
    responses={404: {"description": "Not Found"}},
)


# TODO:: Activate the Apps on go.. And save the App Port when it's deployed
@router.get("/", summary="List All Apps")
async def get_apps():
    apps = list(scan_apps().keys())
    logger.debug(f"APPS: {apps}")
    return apps


@router.get("/{app}", summary="Get More details for an APP that exists in server")
async def get_app(app: str):
    apps = scan_apps()
    if app not in apps:
        raise HTTPException(status_code=404, detail=f"App '{app}' NOT Found")

    logger.debug(f"APP: {apps[app]}")
    return apps[app]["meta"]


@router.put("/{app}", summary="Load a New App to server")
async def load_app(app: str):
    raise HTTPException(status_code=501, detail=f"Not Implemented")


@router.patch("/{app}", summary="Reload existing App")
async def reload_app(app: str):
    apps = scan_apps()
    if app not in apps:
        raise HTTPException(status_code=404, detail=f"App '{app}' NOT Found")
    raise HTTPException(status_code=501, detail=f"Not Implemented")


@router.delete("/{app}", summary="Delete existing App")
async def delete_app(app: str):
    apps = scan_apps()
    if app not in apps:
        raise HTTPException(status_code=404, detail=f"App '{app}' NOT Found")
    raise HTTPException(status_code=501, detail=f"Not Implemented")
