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

"""Sign-in, initial owner setup, and administrator user management."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import Field, SecretStr

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract, User
from monailabel.server.access import Administrator, Principal, Service, request_token
from monailabel.server.demo_access import (
    DemoAccess,
    access,
    admission,
    public_registration_enabled,
    registration_enabled,
)
from monailabel.server.demo_audit import NOTICE
from monailabel.server.demo_gpt import NOTICE as GPT_NOTICE
from monailabel.server.demo_gpt import enabled as gpt_enabled
from monailabel.server.demo_training import preview_enabled

router = APIRouter(prefix="/api/auth")


class Login(Contract):
    username: str = Field(min_length=1, max_length=80)
    password: SecretStr = Field(min_length=1, max_length=256)


def session_cookie(response: Response, token: str, request: Request) -> None:
    response.set_cookie(
        "monailabel_session",
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        max_age=43200,
    )
    response.headers["Cache-Control"] = "no-store"


@router.get("/status")
def status(service: Service) -> dict[str, object]:
    return {
        "setup_required": not service.store.list(User),
        "activity_notice": " ".join(
            note
            for note in (
                NOTICE if service.audit.enabled else "",
                GPT_NOTICE if gpt_enabled() else "",
            )
            if note
        ),
        "training_preview": preview_enabled(),
        "demo_lifecycle": service.demo_lifecycle.enabled,
        "demo_registration": {
            "enabled": True,
            "automatic_approval": public_registration_enabled(),
            **admission(service.store),
        }
        if registration_enabled()
        else {"enabled": False},
    }


@router.post("/setup", status_code=201)
def setup(body: Login, response: Response, request: Request, service: Service) -> User:
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise DomainError("Create the initial administrator from localhost.", status=403)
    user = service.auth.create_user(body.username, body.password.get_secret_value(), bootstrap=True)
    session_cookie(response, service.auth.issue(user), request)
    return user


@router.post("/login")
def login(body: Login, response: Response, request: Request, service: Service) -> User:
    user, token = service.auth.login(
        body.username,
        body.password.get_secret_value(),
        client_address=request.client.host if request.client else "unknown",
    )
    service.audit.record("account.login", user.id)
    session_cookie(response, token, request)
    try:
        service.demo_onboarding.start(user)
    except DomainError:
        # Admission is valid even if a cached sample is temporarily unavailable.
        # The authenticated starter endpoint exposes the actionable preparation error.
        service.audit.record("demo.starter_unavailable", user.id)
    return user


@router.post("/register-demo", status_code=201)
def register_demo(body: Login, response: Response, request: Request, service: Service) -> User:
    if not public_registration_enabled() and (
        not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}
    ):
        raise DomainError(
            "Visitor registration is currently available through the private demo access only.",
            status=403,
        )
    user, token = service.auth.register_demo(body.username, body.password.get_secret_value())
    service.audit.record("account.register", user.id)
    if token:
        session_cookie(response, token, request)
    else:
        response.headers["Cache-Control"] = "no-store"
    return user


@router.get("/me")
def me(user: Principal) -> User:
    return user


@router.post("/signup", status_code=201)
def signup(body: Login, response: Response, request: Request, service: Service) -> User:
    if not public_registration_enabled():
        raise DomainError("Public account registration is not enabled.", status=403)
    return register_demo(body, response, request, service)


@router.get("/session")
def session_info(user: Principal, service: Service) -> dict[str, object]:
    lease = access(service.store, user.id)
    return {
        "server_now": datetime.now(UTC).isoformat(),
        "demo": lease is not None,
        "expires_at": lease.expires_at.isoformat() if lease and lease.expires_at else None,
        "activity_recording": service.audit.enabled,
        "self_registered": bool(lease and lease.self_registered),
        "restricted_roles": bool(lease and lease.restricted_roles),
        "initial_project_id": lease.initial_project_id if lease else None,
    }


@router.get("/accounts")
def accounts(service: Service, user: Administrator, response: Response) -> list[dict[str, object]]:
    """Administrator directory, independent of project membership; never password data."""
    response.headers["Cache-Control"] = "no-store"
    now = datetime.now(UTC)
    result: list[dict[str, object]] = []
    with service.store.transaction() as session:
        leases = {lease.id: lease for lease in session.list(DemoAccess)}
        for account in session.list(User):
            lease = leases.get(account.id)
            status = "Active"
            if not account.active:
                status = "Disabled"
            elif lease and lease.expires_at and lease.expires_at <= now:
                status = "Expired"
            elif lease and not lease.started_at:
                status = "Approved — not started"
            result.append(
                {
                    "id": account.id,
                    "username": account.username,
                    "kind": "Administrator"
                    if account.is_admin
                    else "Demo visitor"
                    if lease
                    else "User",
                    "access": "Annotator + Reviewer"
                    if lease and lease.restricted_roles
                    else "Project roles",
                    "status": status,
                    "expires_at": lease.expires_at.isoformat()
                    if lease and lease.expires_at
                    else None,
                }
            )
    return result


class DemoLogin(Login):
    minutes: Literal[60, 120] = 60


@router.post("/demo-users", status_code=201)
def create_demo_user(body: DemoLogin, user: Administrator, service: Service) -> User:
    return service.auth.create_user(
        body.username, body.password.get_secret_value(), demo_minutes=body.minutes
    )


@router.post("/token")
def token(user: Principal, service: Service) -> dict[str, str]:
    return {"token": service.auth.issue(user)}


@router.post("/logout")
def logout(
    response: Response, request: Request, user: Principal, service: Service
) -> dict[str, bool]:
    value = request_token(request)
    if value:
        service.auth.revoke(value)
    response.delete_cookie("monailabel_session")
    return {"logged_out": True}


@router.get("/users")
def users(user: Administrator, service: Service) -> list[User]:
    return service.store.list(User)


@router.post("/users", status_code=201)
def create_user(body: Login, user: Administrator, service: Service) -> User:
    return service.auth.create_user(body.username, body.password.get_secret_value())


class Active(Contract):
    active: bool


@router.put("/users/{user_id}")
def set_active(user_id: str, body: Active, user: Administrator, service: Service) -> User:
    with service.store.transaction() as session:
        target = session.get(User, user_id)
        if target.id == user.id:
            raise DomainError("You cannot disable your own administrator account.")
        updated = target.model_copy(update={"active": body.active})
        session.update(updated)
    return updated
