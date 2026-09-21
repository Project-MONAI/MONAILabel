"""Authenticated browser desktop pages and a same-origin WebSocket/RFB bridge."""

import asyncio
import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response

from monailabel.core.errors import DomainError
from monailabel.core.models import Asset
from monailabel.server.access import Principal, Service
from monailabel.server.desktops.models import DesktopSession
from monailabel.server.service import Services

router = APIRouter()
STATIC = Path(__file__).parents[1] / "static"
logger = logging.getLogger(__name__)


def summary(desktop: DesktopSession, service: Services) -> dict[str, Any]:
    asset = service.store.get(Asset, desktop.asset_id)
    return {
        "id": desktop.id,
        "viewer": desktop.viewer,
        "mode": desktop.mode,
        "project_id": desktop.project_id,
        "name": asset.name,
        "url": f"/desktop/{desktop.id}",
    }


@router.get("/desktop")
def desktops(user: Principal) -> FileResponse:
    return FileResponse(STATIC / "desktop.html")


@router.get("/api/desktops")
def sessions(service: Service, user: Principal) -> list[dict[str, Any]]:
    result = []
    for desktop in service.store.list(DesktopSession):
        if desktop.user_id == user.id and not desktop.ended:
            # Retain an end action even if a project was removed or access revoked.
            try:
                item = summary(desktop, service)
                service.desktops.owned(desktop.id, user)
            except DomainError:
                item = {
                    "id": desktop.id,
                    "viewer": desktop.viewer,
                    "name": "Unavailable sample",
                    "url": None,
                    "mode": desktop.mode,
                }
            result.append(item)
    return result


@router.get("/api/desktops/{identifier}")
def session_info(identifier: str, service: Service, user: Principal) -> dict[str, Any]:
    return summary(service.desktops.inspect(identifier, user), service)


@router.delete("/api/desktops/{identifier}", status_code=204)
def end_session(identifier: str, service: Service, user: Principal) -> None:
    service.desktops.end(identifier, user)


@router.post("/api/desktops/{identifier}/close-tab", status_code=204)
def close_tab(identifier: str, service: Service, user: Principal) -> None:
    service.desktops.close_tab(identifier, user)


@router.get("/desktop/{identifier}")
def desktop_page(identifier: str, service: Service, user: Principal) -> FileResponse:
    service.desktops.owned(identifier, user)
    return FileResponse(STATIC / "desktop.html")


@router.get("/desktop/{identifier}/client/{path:path}")
def client_resource(identifier: str, path: str, service: Service, user: Principal) -> Response:
    service.desktops.owned(identifier, user)
    root = service.desktops.runtime.dist.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise DomainError("Unknown desktop client resource.", status=404)
    if path == "vnc.html":
        page = candidate.read_text()
        # Replace the upstream inline bootstrap; all executable code stays external.
        page = re.sub(
            r'<script type="module">.*?</script>',
            '<script type="module" src="/static/desktop-client.js"></script>',
            page,
            count=1,
            flags=re.S,
        )
        return HTMLResponse(page)
    return FileResponse(candidate)


@router.websocket("/desktop/{identifier}/socket")
async def display(websocket: WebSocket, identifier: str) -> None:
    service = cast(Services, websocket.app.state.services)
    origin = urlparse(websocket.headers.get("origin", ""))
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    if origin.scheme != scheme or origin.netloc != websocket.headers.get("host"):
        await websocket.close(code=1008)
        return
    token = websocket.cookies.get("monailabel_session")

    def authorize() -> DesktopSession:
        user = service.auth.authenticate(token)
        return service.desktops.owned(identifier, user)

    try:
        desktop = await run_in_threadpool(authorize)
        await run_in_threadpool(service.desktops.renew, desktop)
        reader, writer = await asyncio.open_unix_connection(
            service.desktops.runtime.socket(identifier)
        )
    except (DomainError, OSError):
        await websocket.close(code=1008)
        return

    async def to_browser() -> None:
        while data := await reader.read(65536):
            await websocket.send_bytes(data)

    async def to_desktop() -> None:
        while True:
            data = await websocket.receive_bytes()
            if len(data) > 16 * 1024 * 1024:
                return
            writer.write(data)
            await writer.drain()

    async def check_access() -> None:
        while True:
            await asyncio.sleep(5)
            await run_in_threadpool(authorize)

    tasks: list[asyncio.Task[None]] = []
    connected = False
    try:
        user = await run_in_threadpool(service.auth.authenticate, token)
        await run_in_threadpool(service.desktops.connected, identifier, user)
        connected = True
        await websocket.accept(
            subprotocol="binary" if "binary" in websocket.scope.get("subprotocols", []) else None
        )
        tasks = [asyncio.create_task(c()) for c in (to_browser, to_desktop, check_access)]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except DomainError:
        await websocket.close(code=1008)
    finally:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, (DomainError, WebSocketDisconnect, OSError)
            ):
                logger.error(
                    "Desktop display connection failed",
                    exc_info=(type(result), result, result.__traceback__),
                )
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()
        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.close(code=1000)
        if connected:
            await run_in_threadpool(service.desktops.disconnected, identifier)
