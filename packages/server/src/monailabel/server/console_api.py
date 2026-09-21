"""Web workspace actions, sharing the same services as desktop clients."""

import os
from ipaddress import ip_address
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import Field, SecretStr

from monailabel.core.errors import DomainError
from monailabel.core.models import (
    Annotation,
    Asset,
    AssistantReply,
    AssistantRequest,
    Contract,
    Credential,
    DecisionRequest,
    Job,
    LabelColorsUpdate,
    Membership,
    ModelRecord,
    Project,
    ReviewDecision,
    Role,
    User,
)
from monailabel.server.access import Principal, Service, authorize
from monailabel.server.jobs import JobContext, Outcome
from monailabel.server.labels import update_colors
from monailabel.viewers.manager import ViewerManager

router = APIRouter(prefix="/api", dependencies=[Depends(authorize)])


def local_desktop(request: Request) -> bool:
    def loopback(host: str) -> bool:
        if host == "localhost":
            return True
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    return bool(
        request.client and loopback(request.client.host) and loopback(request.url.hostname or "")
    )


@router.patch("/projects/{project_id}/label-colors")
def label_colors(project_id: str, body: LabelColorsUpdate, service: Service) -> Project:
    return update_colors(service.store, project_id, body)


class CredentialInput(Contract):
    name: str = Field(min_length=1, max_length=120)
    api_key: SecretStr = Field(min_length=1, max_length=16000)
    credential_id: str | None = None


@router.get("/projects/{project_id}/credentials")
def credentials(project_id: str, service: Service) -> list[Credential]:
    return service.store.list(Credential, project_id)


@router.post("/projects/{project_id}/credentials", status_code=201)
def credential(project_id: str, body: CredentialInput, service: Service) -> Credential:
    service.store.get(Project, project_id)
    return service.secrets.save(
        project_id, body.name, body.api_key.get_secret_value(), body.credential_id
    )


class MemberInput(Contract):
    user_id: str | None = None
    username: str | None = Field(default=None, min_length=1, max_length=80)
    roles: list[Role] = Field(min_length=1)


class MemberView(Membership):
    username: str
    active: bool


@router.get("/projects/{project_id}/members")
def members(project_id: str, service: Service) -> list[MemberView]:
    with service.store.transaction() as session:
        result = []
        for member in session.list(Membership, project_id):
            user = session.get(User, member.user_id)
            result.append(
                MemberView(**member.model_dump(), username=user.username, active=user.active)
            )
        return result


@router.put("/projects/{project_id}/members")
def membership(project_id: str, body: MemberInput, service: Service) -> Membership:
    service.store.get(Project, project_id)
    if bool(body.user_id) == bool(body.username):
        raise DomainError("Provide either a username or a user ID.")
    user_id = body.user_id
    if body.username:
        user = next(
            (u for u in service.store.list(User) if u.username == body.username.strip().casefold()),
            None,
        )
        if user is None:
            raise DomainError("User not found. Ask an administrator to create the account first.")
        user_id = user.id
    assert user_id is not None
    return service.auth.add_member(project_id, user_id, body.roles)


@router.get("/projects/{project_id}/permissions")
def permissions(project_id: str, user: Principal, service: Service) -> dict[str, list[str]]:
    return {"roles": sorted(service.auth.roles(user, project_id))}


@router.get("/projects/{project_id}/decisions")
def decisions(project_id: str, service: Service) -> list[ReviewDecision]:
    return service.store.list(ReviewDecision, project_id)


@router.post("/annotations/{annotation_id}/decision", status_code=201)
def decide(
    annotation_id: str, body: DecisionRequest, service: Service, user: Principal
) -> ReviewDecision:
    return service.reviews.decide(annotation_id, body, user)


class ViewerFiles(Contract):
    asset: Asset
    image_path: str | None = None
    mask_path: str | None = None


@router.get("/assets/{asset_id}/viewer-files")
def viewer_files(asset_id: str, request: Request, service: Service) -> ViewerFiles:
    """Resolve the current revision for a viewer sharing this server's filesystem."""
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise DomainError("Local viewer files require a connection on this computer.", status=403)
    asset = service.store.get(Asset, asset_id)
    if asset.kind != "volume3d":
        raise DomainError("Local volume loading requires a 3D sample.")
    annotation = service.store.get(Annotation, asset.annotation_id) if asset.annotation_id else None
    return ViewerFiles(
        asset=asset,
        image_path=str(service.artifacts.path(asset.source_key).resolve())
        if asset.source_key
        else None,
        mask_path=str(service.artifacts.path(annotation.mask_key).resolve())
        if annotation
        else None,
    )


@router.post("/assets/{asset_id}/viewer", status_code=202)
def viewer(
    asset_id: str,
    request: Request,
    service: Service,
    user: Principal,
    name: Literal["slicer", "qupath", "ohif"] | None = None,
    mode: Literal["annotation", "review"] = "annotation",
    target: Literal["auto", "browser"] = "auto",
) -> Job:
    asset = service.store.get(Asset, asset_id)
    if mode == "review":
        service.auth.require(user, asset.project_id, "review")
        if not asset.annotation_id:
            raise DomainError("Submit an annotation before opening it for review.")
    selected = name or ("slicer" if asset.kind == "volume3d" else "qupath")
    if selected == "ohif":
        from monailabel.viewers.ohif import OhifManager

        if asset.kind != "volume3d":
            raise DomainError("Open 2D pathology images in QuPath. OHIF requires a 3D volume.")

        def prepare(context: JobContext) -> Outcome:
            OhifManager().ensure(lambda message: context.progress(0.3, message))
            series = service.dicom.ensure_view(asset_id, context)
            return Outcome(
                {
                    "url": f"/ohif/monailabel?StudyInstanceUIDs={series.study_uid}"
                    f"&assetId={asset.id}&mode={mode}"
                }
            )

        return service.jobs.submit(
            "viewer",
            asset.project_id,
            {"asset_id": asset_id, "viewer": "ohif", "mode": mode},
            prepare,
        )

    if (selected == "slicer") != (asset.kind == "volume3d"):
        raise DomainError("Use Slicer for volumes or QuPath for 2D pathology images.")
    if target == "auto" and local_desktop(request):

        def launch_native(context: JobContext) -> Outcome:
            manager = ViewerManager()
            installation = manager.ensure(
                selected, progress=lambda message: context.progress(0.3, message)
            )
            current = service.store.get(User, user.id)
            if not current.active:
                raise DomainError("Your account is disabled.", status=403)
            service.auth.require(
                current, asset.project_id, "review" if mode == "review" else "read"
            )
            service.store.get(Asset, asset.id)
            token = service.auth.issue(current)
            secret_env = {
                name
                for model in service.store.list(ModelRecord)
                if isinstance(name := model.config.get("token_env"), str)
            }
            return Outcome(
                dict(
                    manager.launch(
                        installation,
                        str(request.base_url).rstrip("/"),
                        asset.project_id,
                        asset_id=asset.id,
                        token=token,
                        secret_env=secret_env,
                        mode=mode,
                        shared_filesystem=selected == "slicer",
                    )
                )
            )

        return service.jobs.submit(
            "viewer",
            asset.project_id,
            {"asset_id": asset_id, "viewer": selected, "mode": mode, "target": "native"},
            launch_native,
        )
    # Native viewers run on this server, including behind an HTTPS reverse proxy.
    server = request.scope.get("server")
    default_url = f"http://127.0.0.1:{server[1]}" if server else str(request.base_url).rstrip("/")
    if getattr(request.app.state, "direct_tls", False):
        default_url = str(request.base_url).rstrip("/")
    url = os.environ.get("MONAILABEL_DESKTOP_BACKEND_URL", default_url).rstrip("/")

    def work(context: JobContext) -> Outcome:
        desktop = service.desktops.open(asset, user, selected, mode, url, context)
        return Outcome({"url": f"/desktop/{desktop.id}", "viewer": selected})

    return service.jobs.submit(
        "viewer", asset.project_id, {"asset_id": asset_id, "viewer": selected, "mode": mode}, work
    )


class WorkspacePrompt(AssistantRequest):
    project_id: str | None = None


@router.post("/assistant")
def workspace_chat(body: WorkspacePrompt, user: Principal, service: Service) -> AssistantReply:
    return service.assistants.run(
        body.project_id,
        AssistantRequest.model_validate(body.model_dump(exclude={"project_id"})),
        user,
    )


@router.get("/assistant/status")
def coordinator_status(service: Service) -> dict[str, str]:
    return (
        service.coordinator.status()
        if service.coordinator
        else {
            "provider": "injected",
            "model": "test",
            "state": "ready",
            "message": "Test provider.",
        }
    )
