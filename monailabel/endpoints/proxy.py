import httpx
from fastapi import APIRouter
from fastapi.responses import Response
import logging

from monailabel.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/proxy",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


@router.get("/dicom/{path:path}", include_in_schema=False)
async def proxy(path: str, response: Response):
    auth = (
        (settings.DICOMWEB_USERNAME, settings.DICOMWEB_PASSWORD)
        if settings.DICOMWEB_USERNAME and settings.DICOMWEB_PASSWORD
        else None
    )

    async with httpx.AsyncClient(auth=auth) as client:
        proxy_path = f"{settings.STUDIES.lstrip('/')}/{path}"
        logger.debug(f"Proxy conneting to {proxy_path}")
        proxy = await client.get(proxy_path)
    response.body = proxy.content
    response.status_code = proxy.status_code
    return response
