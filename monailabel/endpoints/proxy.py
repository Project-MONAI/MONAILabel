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
import time

import google.auth
import google.auth.transport.requests
import requests
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from monailabel.config import settings
from monailabel.endpoints.user.auth import RBAC, User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/proxy",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


async def proxy_dicom(request: Request, op: str, path: str):
    auth = (
        (settings.MONAI_LABEL_DICOMWEB_USERNAME, settings.MONAI_LABEL_DICOMWEB_PASSWORD)
        if settings.MONAI_LABEL_DICOMWEB_USERNAME and settings.MONAI_LABEL_DICOMWEB_PASSWORD
        else None
    )

    headers = {}
    if "googleapis.com" in settings.MONAI_LABEL_STUDIES:
        google_credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        google_credentials.refresh(auth_req)
        token = google_credentials.token
        headers["Authorization"] = "Bearer %s" % token
        auth = None

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
    # print(f"Server {server}; Op: {op}; Prefix: {prefix}; Path: {path}")
    if path.endswith("metadata") and op == "qido":
        prefix = settings.MONAI_LABEL_WADO_PREFIX

    if prefix:
        proxy_path = f"{prefix}/{path}"
    else:
        proxy_path = f"{path}"

    logger.debug(f"Proxy connecting to /dicom/{op}/{path} => {proxy_path}")
    start = time.time()
    if request.method == "POST":
        headers.update(request.headers)
        rp_resp = requests.post(
            f"{server}/{proxy_path}",
            auth=auth,
            stream=True,
            headers=headers,
            data=await request.body(),
        )
    else:
        rp_resp = requests.get(
            f"{server}/{proxy_path}",
            auth=auth,
            stream=True,
            headers=headers,
        )
    logger.debug(f"Proxy Time: {time.time() - start:.4f} => Path: {proxy_path}")

    return Response(
        content=rp_resp.raw.read(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
    )


@router.get("/dicom/wado/{path:path}", include_in_schema=False)
@router.post("/dicom/wado/{path:path}", include_in_schema=False)
async def proxy_wado(
    request: Request,
    path: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    return await proxy_dicom(request, "wado", path)


@router.get("/dicom/qido/{path:path}", include_in_schema=False)
@router.post("/dicom/qido/{path:path}", include_in_schema=False)
async def proxy_qido(
    request: Request,
    path: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    return await proxy_dicom(request, "qido", path)


@router.get("/dicom/stow/{path:path}", include_in_schema=False)
@router.post("/dicom/stow/{path:path}", include_in_schema=False)
async def proxy_stow(
    request: Request,
    path: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    return await proxy_dicom(request, "stow", path)


# https://fastapi.tiangolo.com/tutorial/path-params/#order-matters
@router.get("/dicom/{path:path}", include_in_schema=False)
@router.post("/dicom/{path:path}", include_in_schema=False)
async def proxy(
    request: Request,
    path: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER)),
):
    return await proxy_dicom(request, "", path)
