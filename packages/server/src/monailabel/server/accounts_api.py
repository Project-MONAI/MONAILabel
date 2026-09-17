"""Sign-in, initial owner setup, and administrator user management."""

from fastapi import APIRouter, Request, Response
from pydantic import Field, SecretStr

from monailabel.core.errors import DomainError
from monailabel.core.models import Contract, User
from monailabel.server.access import Administrator, Principal, Service, request_token

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
def status(service: Service) -> dict[str, bool]:
    return {"setup_required": not service.store.list(User)}


@router.post("/setup", status_code=201)
def setup(body: Login, response: Response, request: Request, service: Service) -> User:
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise DomainError("Create the initial administrator from localhost.", status=403)
    user = service.auth.create_user(body.username, body.password.get_secret_value(), bootstrap=True)
    session_cookie(response, service.auth.issue(user), request)
    return user


@router.post("/login")
def login(body: Login, response: Response, request: Request, service: Service) -> User:
    user, token = service.auth.login(body.username, body.password.get_secret_value())
    session_cookie(response, token, request)
    return user


@router.get("/me")
def me(user: Principal) -> User:
    return user


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
