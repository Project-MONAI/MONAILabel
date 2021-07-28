import httpx
from fastapi import APIRouter
from fastapi.responses import Response

from monailabel.config import settings

router = APIRouter(
    prefix="/proxy",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


@router.get("/dicom/{path:path}", include_in_schema=False)
async def proxy(path: str, response: Response):
    async with httpx.AsyncClient() as client:
        proxy = await client.get(f"{settings.DICOM_WEB.lstrip('/')}/{path}")
    response.body = proxy.content
    response.status_code = proxy.status_code
    return response
