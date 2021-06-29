import argparse
import json
import os
import pathlib
import platform
import shutil
import sys
import tempfile

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from monailabel.config import settings
from monailabel.endpoints import activelearning, batch_infer, datastore, download, infer, info, logs, scoring, train
from monailabel.utils.others.generic import get_app_instance, init_log_config

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_STR}/openapi.json",
    docs_url=None,
    redoc_url="/docs",
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

app.include_router(info.router)
app.include_router(infer.router)
app.include_router(batch_infer.router)
app.include_router(train.router)
app.include_router(activelearning.router)
app.include_router(scoring.router)
app.include_router(datastore.router)
app.include_router(download.router)
app.include_router(logs.router)


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace("showExtensions: true,", "showExtensions: true, defaultModelsExpandDepth: -1,")
    return HTMLResponse(body)


@app.on_event("startup")
async def startup_event():
    get_app_instance()


def run_main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    parser_a = subparsers.add_parser("start_server", help="start server help")
    parser_a.add_argument("-a", "--app", required=True, help="App Directory")
    parser_a.add_argument("-s", "--studies", required=True, help="Studies Directory")
    parser_a.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")

    parser_a.add_argument("-i", "--host", default="0.0.0.0", type=str, help="Server IP")
    parser_a.add_argument("-p", "--port", default=8000, type=int, help="Server Port")
    parser_a.add_argument("-l", "--log_config", default=None, type=str, help="Logging config")
    parser_a.add_argument("--dryrun", action="store_true", help="Dry run without starting server")
    parser_a.set_defaults(action="start_server")

    parser_b = subparsers.add_parser("apps", help="sample apps help")
    parser_b.add_argument("-d", "--download", action="store_true", help="download app")
    parser_b.add_argument("-n", "--name", help="Name of the sample app to download", default=None)
    parser_b.add_argument("-o", "--output", help="Output path to save the app", default=None)
    parser_b.set_defaults(action="apps")

    parser_c = subparsers.add_parser("datasets", help="datasets help")
    parser_c.add_argument("-d", "--download", action="store_true", help="download dataset")
    parser_c.add_argument("-n", "--name", help="Name of the dataset to download", default=None)
    parser_c.add_argument("-o", "--output", help="Output path to save the dataset", default=None)
    parser_c.set_defaults(action="datasets")

    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_usage()
        exit(-1)

    if args.action == "apps":
        run_apps(args)
    elif args.action == "datasets":
        run_datasets(args)
    else:
        run_app(args)


def run_datasets(args):
    from monai.apps.datasets import DecathlonDataset
    from monai.apps.utils import download_and_extract

    resource = DecathlonDataset.resource
    md5 = DecathlonDataset.md5

    if not args.download:
        print("Available Datasets are:")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print("  {:<30}: {}".format(k, v))
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
        tarfile_name = f"{dataset_dir}.tar"
        download_and_extract(resource[args.name], tarfile_name, root_dir, md5.get(args.name))

        junk_files = pathlib.Path(dataset_dir).rglob("._*")
        for j in junk_files:
            os.remove(j)
        os.unlink(tarfile_name)
        print(f"{args.name} is downloaded at: {dataset_dir}")


def run_apps(args):
    from monai.apps.utils import download_and_extract

    sample_apps_uri = "https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps"
    resource = {
        "deepedit_brain_tumor": f"{sample_apps_uri}/deepedit_brain_tumor",
        "deepedit_brain_ventricle": f"{sample_apps_uri}/deepedit_brain_ventricle",
        "deepedit_left_atrium": f"{sample_apps_uri}/deepedit_left_atrium",
        "deepedit_lung": f"{sample_apps_uri}/deepedit_lung",
        "deepedit_spleen": f"{sample_apps_uri}/deepedit_spleen",
        "deepedit_vertebra": f"{sample_apps_uri}/deepedit_vertebra",
        "deepgrow_left_atrium": f"{sample_apps_uri}/deepgrow_left_atrium",
        "deepgrow_spleen": f"{sample_apps_uri}/deepgrow_spleen",
        "segmentation_left_atrium": f"{sample_apps_uri}/segmentation_left_atrium",
        "segmentation_spleen": f"{sample_apps_uri}/segmentation_spleen",
    }

    if not args.download:
        print("Available Sample Apps are:")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print("  {:<30}: {}".format(k, v))
    else:
        output_dir = os.path.realpath(os.path.join(args.output, args.name) if args.output else args.name)
        if os.path.exists(output_dir):
            print(f"Directory already exists: {output_dir}")
            exit(-1)

        url = "https://github.com/Project-MONAI/MONAILabel/archive/refs/heads/main.zip"
        tmp_dir = tempfile.mkdtemp(prefix="monailabel")
        tarfile_name = f"{tmp_dir}{os.path.sep}{args.name}.zip"
        download_and_extract(url, tarfile_name, tmp_dir, None)
        app_dir = os.path.join(tmp_dir, "MONAILabel-main", "sample-apps", args.name)

        if os.path.dirname(output_dir):
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        shutil.move(app_dir, output_dir)
        shutil.rmtree(tmp_dir)
        print(f"{args.name} is downloaded at: {output_dir}")


def run_app(args):
    if not os.path.exists(args.app):
        print(f"APP Directory {args.app} NOT Found")
        exit(1)
    if not os.path.exists(args.studies):
        print(f"STUDIES Directory {args.studies} NOT Found")
        exit(1)

    args.app = os.path.realpath(args.app)
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
