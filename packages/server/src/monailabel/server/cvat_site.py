"""Same-origin CVAT viewer with project-scoped access to its private service."""

import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from monailabel.core.errors import DomainError
from monailabel.core.models import ModelRecord, Project, User
from monailabel.core.video import VideoAsset
from monailabel.server.access import Principal, Service
from monailabel.server.service import Services
from monailabel.server.video_editor import VideoEditor

router = APIRouter()
STATIC = Path(__file__).parent / "static"


def binding(
    service: Services, user: User, kind: str, identifier: int, write: bool = False
) -> VideoEditor:
    for editor in reversed(service.store.list(VideoEditor)):
        if editor.ready and getattr(editor, kind + "_id") == identifier:
            service.auth.require(
                user,
                editor.project_id,
                ("review" if editor.mode == "review" else "annotate") if write else "read",
            )
            service.store.get(VideoAsset, editor.asset_id)
            client = service.video_editors.configured()
            if editor.server_url == client.url:
                return editor
    raise DomainError("This CVAT task does not belong to an available workspace video.", status=404)


@router.get("/cvat/editor/{editor_id}")
def panel(editor_id: str, service: Service, user: Principal) -> FileResponse:
    editor = service.store.get(VideoEditor, editor_id)
    service.auth.require(user, editor.project_id)
    if not service.video_editors.managed:
        raise DomainError("This workspace uses an external CVAT viewer.", status=404)
    return FileResponse(STATIC / "cvat.html")


@router.get("/api/cvat/editors/{editor_id}")
def editor_info(editor_id: str, service: Service, user: Principal) -> dict[str, Any]:
    editor = service.store.get(VideoEditor, editor_id)
    service.auth.require(user, editor.project_id)
    asset = service.store.get(VideoAsset, editor.asset_id)
    project = service.store.get(Project, editor.project_id)
    return {
        "editor": editor.model_dump(mode="json"),
        "video": asset.model_dump(mode="json"),
        "project": project.model_dump(mode="json"),
        "roles": sorted(service.auth.roles(user, project.id)),
        "detection_models": [
            {"id": model.id, "name": model.name}
            for model in service.store.list(ModelRecord, project.id)
            if not model.archived and service.video_tracking.supports(model)
        ],
    }


@router.get("/tasks/{task_id}/jobs/{job_id}")
def native(task_id: int, job_id: int, service: Service, user: Principal) -> HTMLResponse:
    editor = binding(service, user, "job", job_id)
    if editor.task_id != task_id or not service.video_editors.managed:
        raise DomainError("Unknown workspace CVAT task.", status=404)
    page = (service.video_editors.manager.dist / "index.html").read_text()
    page = page.replace(
        "<head>",
        '<head><script src="/static/cvat-bridge.js"></script>'
        '<link rel="stylesheet" href="/static/cvat-native.css">',
        1,
    )
    return HTMLResponse(page)


@router.get("/assets/{path:path}")
def resource(path: str, service: Service) -> FileResponse:
    root = (service.video_editors.manager.dist / "assets").resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise DomainError("Unknown CVAT resource.", status=404)
    return FileResponse(candidate)


@router.api_route(
    "/cvat-api/{path:path}", methods=["GET", "HEAD", "POST", "PATCH", "PUT", "DELETE"]
)
async def proxy(path: str, request: Request, service: Service, user: Principal) -> Response:
    if not service.video_editors.managed:
        raise DomainError("The managed CVAT viewer is not enabled.", status=404)
    upstream_path = path
    path = path.rstrip("/")
    method = request.method
    query = dict(request.query_params)
    if query.pop("org", ""):
        raise DomainError("External CVAT organizations are not available.", status=400)
    if query.get("format") == "json":
        query.pop("format")
    empty: dict[str, Any] = {"count": 0, "next": None, "previous": None, "results": []}
    if method == "GET" and path == "users/self":
        return JSONResponse(
            {
                "id": 1,
                "username": user.username,
                "first_name": "",
                "last_name": "",
                "email": "",
                "groups": ["user"],
                "is_staff": False,
                "is_superuser": False,
                "is_active": True,
            }
        )
    if method == "GET" and path in {
        "organizations",
        "invitations",
        "requests",
        "issues",
        "comments",
        "projects",
        "users",
        "growth",
    }:
        return JSONResponse(empty)
    if method in {"GET", "HEAD"} and path in {"lambda/functions", "user-agreements"}:
        return JSONResponse([])
    if method == "GET" and path == "server/plugins":
        return JSONResponse({"GIT_INTEGRATION": False, "ANALYTICS": False, "MODELS": False})
    if method == "POST" and path == "events":
        return JSONResponse({})
    # Never forward caller-controlled URLs, storage locations or organization contexts.
    if set(query) - {
        "id",
        "task_id",
        "job_id",
        "page",
        "page_size",
        "type",
        "quality",
        "number",
        "index",
        "action",
        "scheme",
    }:
        raise DomainError("Unsupported CVAT query.", status=400)
    allowed = method == "GET" and path in {
        "server/about",
        "server/health",
        "server/annotation/formats",
        "server/user-agreements",
        "schema",
    }
    match = re.fullmatch(r"(tasks|jobs)/(\d+)(?:/(annotations|data|data/meta))?", path)
    if match:
        kind, identifier, operation = match.groups()
        permitted = (
            {"type", "quality", "number", "index"}
            if operation == "data"
            else {"action"}
            if operation == "annotations"
            else set()
        )
        if set(query) - permitted:
            raise DomainError("Unsupported CVAT task query.", status=400)
        await run_in_threadpool(binding, service, user, kind[:-1], int(identifier), method != "GET")
        allowed = method == "GET" or (
            kind == "jobs" and operation == "annotations" and method in {"PUT", "PATCH"}
        )
    elif method == "GET" and path in {"labels", "jobs", "tasks"}:
        keys = {"labels": {"task_id", "job_id"}, "tasks": {"id"}, "jobs": {"id", "task_id"}}[path]
        references = set(query) & {"id", "task_id", "job_id"}
        if (
            len(references) > 1
            or references - keys
            or set(query)
            - keys
            - ({"page", "page_size", "type"} if path == "jobs" else {"page", "page_size"})
        ):
            raise DomainError("Choose exactly one workspace task or job.", status=400)
        reference = next(
            ((key, query[key]) for key in ("job_id", "task_id", "id") if key in query), None
        )
        if reference:
            key, identifier = reference
            kind = path[:-1] if key == "id" else key.removesuffix("_id")
            if kind not in {"job", "task"} or not identifier.isdecimal():
                raise DomainError("Choose a workspace CVAT task.", status=400)
            await run_in_threadpool(binding, service, user, kind, int(identifier))
            allowed = True
        else:
            return JSONResponse(empty)
    if not allowed:
        raise DomainError(
            "This CVAT operation is not available in the workspace viewer.", status=403
        )
    body = await request.body()
    if len(body) > 32 * 1024 * 1024:
        raise DomainError("CVAT annotation request exceeds 32 MiB.", status=413)
    client = await run_in_threadpool(service.video_editors.configured)
    try:
        upstream = await run_in_threadpool(
            client.http.request,
            method,
            "/api/" + upstream_path,
            params=query,
            content=body,
            headers={
                "Content-Type": request.headers.get("content-type", "application/json"),
                **{
                    name: request.headers[name]
                    for name in ("range", "if-range")
                    if name in request.headers
                },
            },
        )
    except httpx.HTTPError as exc:
        raise DomainError(
            "The managed CVAT service is unavailable. Your saved draft is preserved.", status=503
        ) from exc
    if upstream.status_code >= 400:
        raise DomainError(
            "CVAT could not complete this operation. Your saved draft is preserved.",
            status=upstream.status_code,
        )
    return Response(
        upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers={
            name: upstream.headers[name]
            for name in (
                "x-checksum",
                "x-updated-date",
                "x-chunk-size",
                "x-media-offset",
                "content-range",
                "accept-ranges",
            )
            if name in upstream.headers
        },
    )
