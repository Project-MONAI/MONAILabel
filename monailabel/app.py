# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from monailabel.config import settings
from monailabel.endpoints import (
    activelearning,
    batch_infer,
    datastore,
    infer,
    info,
    logs,
    ohif,
    proxy,
    scoring,
    session,
    train,
)
from monailabel.interfaces.utils.app import app_instance, clear_cache

app = FastAPI(
    title=settings.MONAI_LABEL_PROJECT_NAME,
    openapi_url="/openapi.json",
    docs_url=None,
    redoc_url="/docs",
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.MONAI_LABEL_CORS_ORIGINS]
            if settings.MONAI_LABEL_CORS_ORIGINS
            else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

static_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "endpoints", "static")
project_root_absolute = pathlib.Path(__file__).parent.parent.resolve()
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(project_root_absolute, "monailabel", "endpoints", "static")),
    name="static",
)

app.include_router(info.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(infer.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(batch_infer.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(train.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(activelearning.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(scoring.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(datastore.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(logs.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(ohif.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(proxy.router, prefix=settings.MONAI_LABEL_API_STR)
app.include_router(session.router, prefix=settings.MONAI_LABEL_API_STR)


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace("showExtensions: true,", "showExtensions: true, defaultModelsExpandDepth: -1,")
    return HTMLResponse(body)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(static_dir, "favicon.ico"), media_type="image/x-icon")


@app.post("/reload", include_in_schema=False)
def reload():
    clear_cache()
    return {}


@app.on_event("startup")
async def startup_event():
    instance = app_instance()
    instance.server_mode(True)
    instance.on_init_complete()
