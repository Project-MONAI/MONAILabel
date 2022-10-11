# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import google.auth
import google.auth.transport.requests
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


class GoogleAuth(httpx.Auth):
    def __init__(self, token):
        self.token = token

    def auth_flow(self, request):
        # Send the request, with a custom `Authorization` header.
        request.headers["Authorization"] = "Bearer %s" % self.token
        yield request


async def proxy_dicom(op: str, path: str, response: Response):
    auth = (
        (settings.MONAI_LABEL_DICOMWEB_USERNAME, settings.MONAI_LABEL_DICOMWEB_PASSWORD)
        if settings.MONAI_LABEL_DICOMWEB_USERNAME and settings.MONAI_LABEL_DICOMWEB_PASSWORD
        else None
    )
    if "googleapis.com" in settings.MONAI_LABEL_STUDIES:
        google_credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        google_credentials.refresh(auth_req)
        token = google_credentials.token
        auth = GoogleAuth(token)

    async with httpx.AsyncClient(auth=auth) as client:
        server = f"{settings.MONAI_LABEL_STUDIES.rstrip('/')}"
        prefix = (
            settings.MONAI_LABEL_WADO_PREFIX
            if op == "wado"
            else settings.MONAI_LABEL_QIDO_PREFIX
            if op == "qido"
            else settings.MONAI_LABEL_STOW_PREFIX
            if op == "stow"
            else ""
        )

        # some version of ohif requests metadata using qido so change it to wado
        if path.endswith("metadata") and op == "qido":
            prefix = settings.MONAI_LABEL_WADO_PREFIX

        if prefix:
            proxy_path = f"{server}/{prefix}/{path}"
        else:
            proxy_path = f"{server}/{path}"

        logger.debug(f"Proxy connecting to /dicom/{op}/{path} => {proxy_path}")
        timeout = httpx.Timeout(
            settings.MONAI_LABEL_DICOMWEB_PROXY_TIMEOUT,
            read=settings.MONAI_LABEL_DICOMWEB_READ_TIMEOUT,
        )
        proxy = await client.get(proxy_path, timeout=timeout)

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


# https://fastapi.tiangolo.com/tutorial/path-params/#order-matters
@router.get("/dicom/{path:path}", include_in_schema=False)
async def proxy(path: str, response: Response, user: User = Depends(get_basic_user)):
    return await proxy_dicom("", path, response)
