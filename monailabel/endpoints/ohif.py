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
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from monailabel.config import settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.utils.others.generic import get_mime_type

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ohif",
    tags=["Others"],
    responses={404: {"description": "Not found"}},
)


OHIF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static", "ohif")


def _ohif_file(ohif_dir: str, path: str) -> str:
    """Existing file inside ``ohif_dir`` for the requested path, else the viewer's index page (client-side routes)."""
    index = os.path.join(ohif_dir, "index.html")
    if not path:
        return index

    try:
        candidate = os.path.realpath(os.path.join(ohif_dir, path))
        inside = os.path.commonpath([candidate, ohif_dir]) == ohif_dir
    except ValueError:
        # e.g. embedded null byte, or a different drive on Windows
        return index
    return candidate if inside and os.path.isfile(candidate) else index


def get_ohif(path: str):
    ohif_dir = os.path.realpath(OHIF_DIR)
    file = _ohif_file(ohif_dir, path)

    if not os.path.exists(file):
        logger.info(file)
        raise HTTPException(status_code=404, detail="Resource NOT Found")

    return FileResponse(file, media_type=get_mime_type(file))

    # headers = {
    #     "Cross-Origin-Opener-Policy": "same-origin",
    #     "Cross-Origin-Embedder-Policy": "require-corp",
    #     "Cross-Origin-Resource-Policy": "same-site",
    # }

    # return FileResponse(file, media_type=get_mime_type(file), headers=headers)


@router.get("/{path:path}", include_in_schema=False)
async def api_get_ohif(path: str, user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))):
    return get_ohif(path)
