import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from server.core.config import settings
from server.endpoints import activelearning, inference, logs, train, apps, dataset
from server.utils.generic import init_log_config

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_STR}/openapi.json",
    docs_url=None,
    redoc_url="/docs"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(inference.router)
app.include_router(train.router)
app.include_router(activelearning.router)
app.include_router(apps.router)
app.include_router(dataset.router)
app.include_router(logs.router)


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace('showExtensions: true,', 'showExtensions: true, defaultModelsExpandDepth: -1,')
    return HTMLResponse(body)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workspace', required=True)
    parser.add_argument('-d', '--debug', action='store_true')

    parser.add_argument('-i', '--host', default="127.0.0.1", type=str)
    parser.add_argument('-p', '--port', default=8000, type=int)
    parser.add_argument('-r', '--reload', action='store_true')
    parser.add_argument('-l', '--log_config', default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.workspace, exist_ok=True)
    args.workspace = os.path.realpath(args.workspace)

    for arg in vars(args):
        logger.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    # Prepare the workspace
    settings.WORKSPACE = args.workspace
    os.putenv("WORKSPACE", args.workspace)

    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(os.path.join(args.workspace, "apps"), exist_ok=True)
    os.makedirs(os.path.join(args.workspace, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(args.workspace, "logs"), exist_ok=True)

    uvicorn.run(
        "main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        reload=args.reload,
        log_config=init_log_config(args.log_config, args.workspace, "server.log"),
        use_colors=True,
    )


if __name__ == '__main__':
    run_main()
