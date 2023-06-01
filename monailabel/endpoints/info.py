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
from typing import List

from fastapi import APIRouter, Depends

from monailabel.config import RBAC_ADMIN, RBAC_USER, settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance

router = APIRouter(
    prefix="/info",
    tags=["App"],
    responses={404: {"description": "Not found"}},
)


def app_info():
    instance: MONAILabelApp = app_instance()
    return instance.info()


@router.get("/", summary=f"{RBAC_USER}Get App Info")
async def api_app_info(user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))):
    return app_info()


@router.post("/multinode", summary=f"{RBAC_ADMIN}Enable multi-node support (Add other hosts)")
async def api_app_multinode(hostnames: List[str], user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ADMIN))):
    for hostname in hostnames:
        h = hostname.rstrip("/")
        if h not in settings.MONAI_LABEL_HOSTS:
            settings.MONAI_LABEL_HOSTS.append(h)
    return settings.MONAI_LABEL_HOSTS


@router.delete("/multinode", summary=f"{RBAC_ADMIN}Disable multi-node support (Remove all hosts)")
async def api_app_multinode_remove(user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_ADMIN))):
    settings.MONAI_LABEL_HOSTS.clear()
    return settings.MONAI_LABEL_HOSTS
