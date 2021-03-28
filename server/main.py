import json
import logging
import os

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from server.core.config import settings
from server.endpoints import activelearning, inference, logs, session, train, apps, dataset

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
app.include_router(session.router)
app.include_router(logs.router)

if os.path.exists('logging.json'):
    with open('logging.json', 'rt') as f:
        config = json.load(f)
    print('Initializing Logging from config file...')
    logging.config.dictConfig(config)
else:
    print('Initializing Default Logging...')
    logging.basicConfig(
        level=(logging.INFO),
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace('showExtensions: true,', 'showExtensions: true, defaultModelsExpandDepth: -1,')
    return HTMLResponse(body)
