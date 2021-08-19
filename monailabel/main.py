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

import argparse
import json
import os
import pathlib
import platform
import shutil
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from monailabel.config import settings
from monailabel.endpoints import activelearning, batch_infer, datastore, infer, info, logs, scoring, train
from monailabel.utils.others.app_utils import app_instance
from monailabel.utils.others.generic import init_log_config

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS] if settings.CORS_ORIGINS else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_STR}/openapi.json",
    docs_url=None,
    redoc_url="/docs",
    middleware=middleware,
)

static_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
app.mount(
    "/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")), name="static"
)

app.include_router(info.router)
app.include_router(infer.router)
app.include_router(batch_infer.router)
app.include_router(train.router)
app.include_router(activelearning.router)
app.include_router(scoring.router)
app.include_router(datastore.router)
app.include_router(logs.router)


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace("showExtensions: true,", "showExtensions: true, defaultModelsExpandDepth: -1,")
    return HTMLResponse(body)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(static_dir, "favicon.ico"), media_type="image/x-icon")


@app.on_event("startup")
async def startup_event():
    app_instance()


def run_main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    parser_a = subparsers.add_parser("start_server", help="start server for monailabel")
    parser_a.add_argument("-a", "--app", required=True, help="App Directory")
    parser_a.add_argument("-s", "--studies", required=True, help="Studies Directory")
    parser_a.add_argument("-u", "--username", required=False, default=None, help="Username to access DICOMWeb server")
    parser_a.add_argument("-w", "--password", required=False, default=None, help="Password to access DICOMWeb server")
    parser_a.add_argument("-W", "--wado_prefix", required=False, default="", help="DICOMWeb Server WADO URL prefix")
    parser_a.add_argument("-Q", "--qido_prefix", required=False, default="", help="DICOMWeb Server QIDO URL prefix")
    parser_a.add_argument("-S", "--stow_prefix", required=False, default="", help="DICOMWeb Server STOW URL prefix")
    parser_a.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")

    parser_a.add_argument("-i", "--host", default="0.0.0.0", type=str, help="Server IP")
    parser_a.add_argument("-p", "--port", default=8000, type=int, help="Server Port")
    parser_a.add_argument("-l", "--log_config", default=None, type=str, help="Logging config")
    parser_a.add_argument("--dryrun", action="store_true", help="Dry run without starting server")
    parser_a.set_defaults(action="start_server")

    parser_b = subparsers.add_parser("apps", help="list or download sample apps")
    parser_b.add_argument("-d", "--download", action="store_true", help="download app")
    parser_b.add_argument("-n", "--name", help="Name of the sample app to download", default=None)
    parser_b.add_argument("-o", "--output", help="Output path to save the app", default=None)
    parser_b.add_argument("--prefix", default=None)
    parser_b.set_defaults(action="apps")

    parser_c = subparsers.add_parser("datasets", help="list or download sample datasets")
    parser_c.add_argument("-d", "--download", action="store_true", help="download dataset")
    parser_c.add_argument("-n", "--name", help="Name of the dataset to download", default=None)
    parser_c.add_argument("-o", "--output", help="Output path to save the dataset", default=None)
    parser_c.add_argument("--prefix", default=None)
    parser_c.set_defaults(action="datasets")

    parser_d = subparsers.add_parser("plugins", help="list or download viewer plugins")
    parser_d.add_argument("-d", "--download", action="store_true", help="download plugin")
    parser_d.add_argument("-n", "--name", help="Name of the plugin to download", default=None)
    parser_d.add_argument("-o", "--output", help="Output path to save the plugin", default=None)
    parser_d.add_argument("--prefix", default=None)
    parser_d.set_defaults(action="plugins")

    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_usage()
        exit(-1)

    if args.action == "apps":
        action_apps(args)
    elif args.action == "datasets":
        action_datasets(args)
    elif args.action == "plugins":
        action_plugins(args)
    else:
        run_app(args)


def action_datasets(args):
    from monai.apps.datasets import DecathlonDataset
    from monai.apps.utils import download_and_extract

    resource = DecathlonDataset.resource
    md5 = DecathlonDataset.md5

    if not args.download:
        print("Available Datasets are:")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print("  {:<30}: {}".format(k, v))
        print("")
    else:
        url = resource.get(args.name) if args.name else None
        if not url:
            print(f"Dataset ({args.name}) NOT Exists.")

            available = "  " + "\n  ".join(resource.keys())
            print(f"Available Datasets are:: \n{available}")
            print("----------------------------------------------------")
            exit(-1)

        dataset_dir = os.path.join(args.output, args.name) if args.output else args.name
        if os.path.exists(dataset_dir):
            print(f"Directory already exists: {dataset_dir}")
            exit(-1)

        root_dir = os.path.dirname(os.path.realpath(dataset_dir))
        os.makedirs(root_dir, exist_ok=True)
        tarfile_name = f"{dataset_dir}.tar"
        download_and_extract(resource[args.name], tarfile_name, root_dir, md5.get(args.name))

        junk_files = pathlib.Path(dataset_dir).rglob("._*")
        for j in junk_files:
            os.remove(j)
        os.unlink(tarfile_name)
        print(f"{args.name} is downloaded at: {dataset_dir}")


def action_apps(args):
    apps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "sample-apps")
    if not os.path.exists(apps_dir):
        apps_dir = os.path.join(args.prefix if args.prefix else sys.prefix, "monailabel", "sample-apps")

    apps = os.listdir(apps_dir)
    apps = [os.path.basename(a) for a in apps]
    apps.sort()

    resource = {
        "Template/Generic Apps": [a for a in apps if a.startswith("generic_")],
        "Deepedit based Apps": [a for a in apps if a.startswith("deepedit_")],
        "Deepgrow based Apps": [a for a in apps if a.startswith("deepgrow_")],
        "Standard Segmentation Apps": [a for a in apps if a.startswith("segmentation_")],
    }

    if not args.download:
        print(f"Available Sample Apps are: ({apps_dir})")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print(f"{k}")
            print("----------------------------------------------------")
            for n in v:
                print("  {:<30}: {}".format(n, f"{apps_dir}/{n}"))
            print("")
    else:
        app_dir = os.path.join(apps_dir, args.name)
        if args.name not in apps or not os.path.exists(apps_dir):
            print(f"App {args.name} => {app_dir} not exists")
            exit(-1)

        output_dir = os.path.realpath(os.path.join(args.output, args.name) if args.output else args.name)
        if os.path.exists(output_dir):
            print(f"Directory already exists: {output_dir}")
            exit(-1)

        if os.path.dirname(output_dir):
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        shutil.copytree(app_dir, output_dir, ignore=shutil.ignore_patterns("logs", "model", "__pycache__"))
        print(f"{args.name} is copied at: {output_dir}")


def action_plugins(args):
    plugins_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "plugins")
    if not os.path.exists(plugins_dir):
        plugins_dir = os.path.join(args.prefix if args.prefix else sys.prefix, "monailabel", "plugins")

    plugins = os.listdir(plugins_dir)
    plugins = [os.path.basename(a) for a in plugins]
    plugins = [p for p in plugins if p != "ohif"]
    plugins.sort()

    resource = {p: f"{plugins_dir}/{p}" for p in plugins}

    if not args.download:
        print("Available Plugins are:")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print("  {:<30}: {}".format(k, v))
        print("")
    else:
        plugin_dir = os.path.join(plugins_dir, args.name)
        if args.name not in plugins or not os.path.exists(plugin_dir):
            print(f"Plugin {args.name} => {plugins_dir} not exists")
            exit(-1)

        output_dir = os.path.realpath(os.path.join(args.output, args.name) if args.output else args.name)
        if os.path.exists(output_dir):
            print(f"Directory already exists: {output_dir}")
            exit(-1)

        if os.path.dirname(output_dir):
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        shutil.copytree(plugin_dir, output_dir, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"{args.name} is copied at: {output_dir}")


def run_app(args):
    if not os.path.exists(args.app):
        print(f"APP Directory {args.app} NOT Found")
        exit(1)
    if (
        not args.studies.startswith("http://")
        and not args.studies.startswith("https://")
        and not os.path.exists(args.studies)
    ):
        print(f"STUDIES Directory {args.studies} NOT Found")
        exit(1)

    args.app = os.path.realpath(args.app)
    if not args.studies.startswith("http://") and not args.studies.startswith("https://"):
        args.studies = os.path.realpath(args.studies)

    for arg in vars(args):
        print("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    overrides = {
        "APP_DIR": args.app,
        "STUDIES": args.studies,
    }
    for k, v in overrides.items():
        os.putenv(k, str(v))

    settings.APP_DIR = args.app
    settings.STUDIES = args.studies
    settings.DICOMWEB_USERNAME = args.username
    settings.DICOMWEB_PASSWORD = args.password
    settings.QIDO_PREFIX = args.qido_prefix
    settings.WADO_PREFIX = args.wado_prefix
    settings.STOW_PREFIX = args.stow_prefix

    dirs = ["model", "lib", "logs"]
    for d in dirs:
        d = os.path.join(args.app, d)
        if not os.path.exists(d):
            os.makedirs(d)

    sys.path.append(args.app)
    sys.path.append(os.path.join(args.app, "lib"))

    if args.dryrun:
        with open(".env", "w") as f:
            for k, v in settings.dict().items():
                v = json.dumps(v) if isinstance(v, list) else v
                e = f"{k}={v}"
                f.write(e)
                f.write(os.linesep)
                print(f"{'set' if any(platform.win32_ver()) else 'export'} {e}")
    else:
        print("")
        print("**********************************************************")
        print("                  ENV VARIABLES/SETTINGS                  ")
        print("**********************************************************")
        for k, v in settings.dict().items():
            v = json.dumps(v) if isinstance(v, list) else v
            print(f"{'set' if any(platform.win32_ver()) else 'export'} {k}={v}")
        print("**********************************************************")
        print("")

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug" if args.debug else "info",
            log_config=init_log_config(args.log_config, args.app, "app.log"),
            use_colors=True,
            access_log=args.debug,
        )

    sys.path.remove(os.path.join(args.app, "lib"))
    sys.path.remove(args.app)
    for k in overrides:
        os.unsetenv(k)


if __name__ == "__main__":
    run_main()
