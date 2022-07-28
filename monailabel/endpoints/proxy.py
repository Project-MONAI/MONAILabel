import logging

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from monailabel.config import settings
from monailabel.endpoints.user.auth import User, get_basic_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/proxy",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


async def proxy_dicom(op: str, path: str, response: Response):
    auth = (
        (settings.MONAI_LABEL_DICOMWEB_USERNAME, settings.MONAI_LABEL_DICOMWEB_PASSWORD)
        if settings.MONAI_LABEL_DICOMWEB_USERNAME and settings.MONAI_LABEL_DICOMWEB_PASSWORD
        else None
    )

    async with httpx.AsyncClient(auth=auth) as client:
        server = f"{settings.MONAI_LABEL_STUDIES.rstrip('/')}"
        prefix = (
            settings.MONAI_LABEL_WADO_PREFIX
            if op == "wado"
            else settings.MONAI_LABEL_QIDO_PREFIX
            if op == "qido"
            else settings.MONAI_LABEL_STOW_PREFIX
        )

        # some version of ohif requests metadata using qido so change it to wado
        if path.endswith("metadata") and op == "qido":
            prefix = settings.MONAI_LABEL_WADO_PREFIX

        if prefix:
            proxy_path = f"{server}/{prefix}/{path}"
        else:
            proxy_path = f"{server}/{path}"

        logger.debug(f"Proxy connecting to /dicom/{op}/{path} => {proxy_path}")
        proxy = await client.get(proxy_path)
    response.body = proxy.content
    response.status_code = proxy.status_code
    return response


@router.get("/dicom/wado/{path:path}", include_in_schema=False)
async def proxy_wado(path: str, response: Response, user: User = Depends(get_basic_user)):
    return await proxy_dicom("wado", path, response)


@router.get("/dicom/qido/{path:path}", include_in_schema=False)
async def proxy_qido(path: str, response: Response, user: User = Depends(get_basic_user)):
    return await proxy_dicom("qido", path, response)


@router.get("/dicom/stow/{path:path}", include_in_schema=False)
async def proxy_stow(path: str, response: Response, user: User = Depends(get_basic_user)):
    return await proxy_dicom("stow", path, response)
