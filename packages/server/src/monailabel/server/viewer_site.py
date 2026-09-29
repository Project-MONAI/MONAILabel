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

"""Serve the built OHIF application on the same origin as authenticated APIs."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from monailabel.core.errors import DomainError
from monailabel.viewers.ohif import OhifManager

router = APIRouter()


@router.get("/ohif/{path:path}")
def ohif(path: str) -> FileResponse:
    root = OhifManager().dist.resolve()
    if not (root / "index.html").exists():
        raise DomainError("Prepare OHIF first: uv run monailabel viewer ohif", status=503)
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise DomainError("Unknown viewer resource.", status=404)
    if not candidate.is_file():
        if candidate.suffix:
            raise DomainError("Unknown viewer resource.", status=404)
        candidate = root / "index.html"
    return FileResponse(candidate)
