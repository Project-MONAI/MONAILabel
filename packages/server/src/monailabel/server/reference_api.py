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

"""Bounded multipart transport for an image and its external reference labels."""

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.types import Message

from monailabel.core.errors import DomainError
from monailabel.core.reference_imports import (
    EvaluationImportRequest,
    EvaluationImportResult,
    LabelImportRequest,
)
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.data import MAX_FILE_BYTES

router = APIRouter(prefix="/api/projects/{project_id}", dependencies=[Depends(authorize)])


@router.post("/evaluation-imports", status_code=201)
@router.post("/label-imports", status_code=201)
async def import_reference(
    project_id: str, request: Request, service: Service, user: Principal
) -> EvaluationImportResult:
    size = 0

    async def receive() -> Message:
        nonlocal size
        message = await request.receive()
        size += len(message.get("body", b""))
        if size > MAX_FILE_BYTES * 2 + 65536:
            raise DomainError("Image and label files must each be at most 128 MiB.", status=413)
        return message

    bounded = Request(request.scope, receive=receive)
    async with bounded.form(max_files=2, max_fields=1, max_part_size=65536) as form:
        metadata, image, labels = form.get("metadata"), form.get("image"), form.get("labels")
        if (
            not isinstance(metadata, str)
            or not isinstance(image, UploadFile)
            or not isinstance(labels, UploadFile)
        ):
            raise DomainError("Choose an image, its label file and import settings.")
        try:
            contract = (
                EvaluationImportRequest
                if request.url.path.endswith("/evaluation-imports")
                else LabelImportRequest
            )
            body = contract.model_validate_json(metadata)
        except ValidationError as exc:
            raise DomainError(str(exc.errors()[0]["msg"])) from exc
        if body.reviewed:
            service.auth.require(user, project_id, "review")
        content, mask = await image.read(MAX_FILE_BYTES + 1), await labels.read(MAX_FILE_BYTES + 1)
        return await run_in_threadpool(
            service.reference_imports.import_pair,
            project_id,
            body,
            image.filename or "",
            content,
            labels.filename or "",
            mask,
            user.id,
        )
