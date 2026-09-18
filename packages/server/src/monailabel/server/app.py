import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from monailabel.core.chat import ChatProvider
from monailabel.core.errors import DomainError
from monailabel.providers.chat.config import CoordinatorConfig
from monailabel.server.accounts_api import router as accounts_router
from monailabel.server.api import router
from monailabel.server.console_api import router as console_router
from monailabel.server.cvat_site import router as cvat_router
from monailabel.server.dicom_api import router as dicom_router
from monailabel.server.evaluation_api import router as evaluation_router
from monailabel.server.reference_api import router as reference_router
from monailabel.server.service import Services
from monailabel.server.video_api import router as video_router
from monailabel.server.viewer_site import router as viewer_router
from monailabel.server.workspace import workspace_dir


def create_app(
    data_dir: Path | None = None,
    *,
    chat_provider: ChatProvider | None = None,
    coordinator: CoordinatorConfig | None = None,
) -> FastAPI:
    directory = data_dir or workspace_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = Services(
            directory, chat_provider=chat_provider, coordinator=coordinator
        )
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(title="MONAI Label", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.environ.get(
            "MONAILABEL_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver"
        ).split(","),
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            parsed = urlparse(origin)
            if parsed.netloc != request.headers.get("host") or parsed.scheme != request.url.scheme:
                return JSONResponse(
                    status_code=403, content={"detail": "Cross-origin writes are disabled."}
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "img-src 'self' https://fastapi.tiangolo.com; frame-ancestors 'none'"
            )
        if request.url.path.startswith("/ohif/"):
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "connect-src 'self' blob:; worker-src 'self' blob:; font-src 'self' data:; "
                "frame-ancestors 'none'"
            )
        if request.url.path.startswith(("/tasks/", "/cvat/", "/assets/")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "connect-src 'self' blob:; worker-src 'self' blob:; font-src 'self' data:; "
                "frame-src 'self'; frame-ancestors 'self'"
            )
        if request.url.path.startswith(("/api", "/cvat-api", "/tasks/", "/cvat/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"code": exc.code, "detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Validation errors must never echo password or API-key input values.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
            },
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    for page in ("datasets", "models", "reviews", "activity", "team"):
        app.add_api_route(f"/{page}", index, methods=["GET"], include_in_schema=False)

    @app.get("/static/speech.js", include_in_schema=False)
    def speech() -> Response:
        source = files("monailabel.viewers").joinpath("resources/ohif/extension/src/speech.js")
        return Response(source.read_text(), media_type="text/javascript")

    app.include_router(accounts_router)
    app.include_router(router)
    app.include_router(console_router)
    app.include_router(dicom_router)
    app.include_router(evaluation_router)
    app.include_router(reference_router)
    app.include_router(viewer_router)
    app.include_router(video_router)
    app.include_router(cvat_router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
