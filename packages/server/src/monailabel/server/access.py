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

"""HTTP authentication and centralized resource access checks."""

from typing import Annotated, cast

from fastapi import Depends, Request

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Annotation,
    Asset,
    ClassificationProposal,
    Evaluation,
    Job,
    Promotion,
    Proposal,
    RegionProposal,
    User,
)
from monailabel.core.review_units import ReviewUnit
from monailabel.core.video import VideoAsset
from monailabel.server.service import Services


def services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


Service = Annotated[Services, Depends(services)]


def request_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    return (
        authorization.removeprefix("Bearer ")
        if authorization.startswith("Bearer ")
        else request.cookies.get("monailabel_session")
    )


def principal(request: Request, service: Service) -> User:
    return service.auth.authenticate(request_token(request))


Principal = Annotated[User, Depends(principal)]


def authorize(request: Request, service: Service, user: Principal) -> None:
    params = request.path_params
    project_id = params.get("project_id")
    for key, model in (
        ("unit_id", ReviewUnit),
        ("video_id", VideoAsset),
        ("asset_id", Asset),
        ("annotation_id", Annotation),
        ("proposal_id", Proposal),
        ("evaluation_id", Evaluation),
        ("promotion_id", Promotion),
        ("job_id", Job),
        ("region_id", RegionProposal),
        ("classification_id", ClassificationProposal),
    ):
        if key in params:
            record = service.store.get(model, params[key])
            if isinstance(
                record,
                (
                    Asset,
                    Annotation,
                    Proposal,
                    Evaluation,
                    Promotion,
                    Job,
                    RegionProposal,
                    ClassificationProposal,
                    VideoAsset,
                    ReviewUnit,
                ),
            ):
                project_id = record.project_id
            break
    if project_id is None:
        return
    action = "read"
    if request.method not in {"GET", "HEAD"}:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint in {
            "review",
            "review-mask",
            "restore",
            "annotate",
            "viewer-inference",
            "spatial-inference",
            "reject",
            "cancel",
            "mask-import",
            "region-submissions",
        }:
            action = "annotate"
        elif endpoint in {"decision", "review-complete", "review-decisions"}:
            action = "review"
        elif endpoint in {
            "label-colors",
            "cvat-submit",
            "track",
            "find-and-track",
            "interactive",
            "target",
        }:
            action = "edit"
        elif endpoint not in {"assistant", "viewer", "editor", "exports"}:
            action = "manage"
    if request.url.path.endswith(("/credentials", "/members", "/model-services")):
        action = "manage"
    service.auth.require(user, project_id, action)


def administrator(user: Principal) -> User:
    if not user.is_admin:
        raise DomainError("Administrator access is required.", status=403)
    return user


Administrator = Annotated[User, Depends(administrator)]
