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

"""Service-level capabilities for leased visitors, independent of public signup."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from monailabel.core.errors import DomainError
from monailabel.core.models import LearnerCreate, Membership, ModelRecord, Role, User
from monailabel.server.demo_access import access, require_active
from monailabel.server.demo_context import actor
from monailabel.server.storage import Session, Store

# Only verified template-import jobs and fixed synthetic fixtures enter this context.
# No HTTP/tool contract accepts a bypass flag. Bind it to one actor/project.
_prepared_import: ContextVar[tuple[str | None, str] | None] = ContextVar(
    "monailabel_prepared_import", default=None
)


def visitor(store: Store | Session, user_id: str | None = None) -> bool:
    identifier = user_id if user_id is not None else actor.get()
    return bool(identifier and access(store, identifier))


def require_operator(store: Store | Session, operation: str) -> None:
    if visitor(store):
        raise DomainError(
            f"{operation} is managed by the demo host. Use the prepared demo choices.",
            code="demo_restricted",
            status=403,
        )


def setup_allowed(store: Store | Session, project_id: str, user_id: str | None = None) -> bool:
    identifier = user_id if user_id is not None else actor.get()
    lease = access(store, identifier) if identifier else None
    return bool(
        lease
        and lease.restricted_roles
        and lease.initial_project_id == project_id
        and any(
            member.user_id == identifier and Role.ANNOTATOR in member.roles
            for member in store.list(Membership, project_id)
        )
    )


def require_setup(store: Store, project_id: str, request: LearnerCreate) -> None:
    """A visitor may create fixed local setups, never supply connections or weights."""
    if not visitor(store):
        return
    identifier = actor.get()
    assert identifier is not None
    require_active(store, store.get(User, identifier))
    if not setup_allowed(store, project_id):
        raise DomainError("Create demo model setups only in your own demo project.", status=403)
    if request.config or request.recipe not in {
        "monai-unet",
        "vista3d",
        "totalsegmentator-ct",
        "totalsegmentator-mr",
    }:
        raise DomainError(
            "Demo setups use the prepared model types and default settings.", status=403
        )
    if request.recipe == "monai-unet":
        if request.initial_model_id:
            raise DomainError("The demo U-Net setup starts from scratch.", status=403)
    else:
        parent = store.get(ModelRecord, request.initial_model_id or "")
        if (
            parent.project_id != project_id
            or parent.archived
            or not parent.read_only
            or parent.preset != request.recipe
            or parent.provider != request.recipe
        ):
            raise DomainError("Choose the matching prepared base model.", status=403)


def require_project_creator(store: Store | Session) -> None:
    identifier = actor.get()
    lease = access(store, identifier) if identifier else None
    if lease and lease.restricted_roles:
        raise DomainError(
            "Your demo includes a private prepared project with Annotator and Reviewer access.",
            code="demo_restricted",
            status=403,
        )


def require_prepared_import(store: Store | Session, project_id: str) -> None:
    if _prepared_import.get() != (actor.get(), project_id):
        require_operator(store, "Dataset upload")


@contextmanager
def prepared_import(project_id: str) -> Iterator[None]:
    token = _prepared_import.set((actor.get(), project_id))
    try:
        yield
    finally:
        _prepared_import.reset(token)
