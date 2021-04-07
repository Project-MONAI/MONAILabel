import subprocess

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(
    prefix="/tools",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


# TODO:: Provide tools to download sample pipeline template etc..
#  Think what else can go here.. that can be helpful to users...

@router.get("/app_template", summary="Download template to develop a new App")
async def app_template():
    raise HTTPException(status_code=501, detail=f"Not Implemented")


@router.get("/gpu_info", summary="Get GPU Info (nvidia-smi)")
async def gpu_info():
    response = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    return Response(content=response, media_type='text/plain')
