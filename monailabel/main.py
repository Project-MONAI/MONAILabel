import argparse
import json
import os
import pathlib
import platform
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from monailabel.config import settings
from monailabel.endpoints import activelearning, batch_infer, datastore, download, inference, info, logs, train
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
app.include_router(inference.router)
app.include_router(batch_infer.router)
app.include_router(train.router)
app.include_router(activelearning.router)
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

    parser_a = subparsers.add_parser("run", help="run help")
    parser_a.add_argument("-a", "--app", required=True, help="App Directory")
    parser_a.add_argument("-s", "--studies", required=True, help="Studies Directory")
    parser_a.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")

    parser_a.add_argument("-i", "--host", default="0.0.0.0", type=str, help="Server IP")
    parser_a.add_argument("-p", "--port", default=8000, type=int, help="Server Port")
    parser_a.add_argument("-l", "--log_config", default=None, type=str, help="Logging config")
    parser_a.add_argument("--dryrun", action="store_true", help="Dry run without starting server")
    parser_a.set_defaults(action="run")

    parser_b = subparsers.add_parser("samples", help="samples help")
    parser_b.add_argument("-c", "--command", choices=["list", "download"], help="list or download samples")
    parser_b.add_argument("-n", "--name", help="Name of the sample to download", default=None)
    parser_b.set_defaults(action="samples")

    parser_c = subparsers.add_parser("datasets", help="datasets help")
    parser_c.add_argument("-c", "--command", choices=["list", "download"], help="list or download datasets")
    parser_c.add_argument("-n", "--name", help="Name of the dataset to download", default=None)
    parser_c.add_argument("-o", "--output", help="Output path to save the dataset", default=None)
    parser_c.set_defaults(action="datasets")

    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_usage()
        exit(-1)

    if args.action == "samples":
        run_samples(args)
    elif args.action == "datasets":
        run_datasets(args)
    else:
        run_app(args)


def run_datasets(args):
    from monai.apps.datasets import DecathlonDataset
    from monai.apps.utils import download_and_extract

    resource = DecathlonDataset.resource
    md5 = DecathlonDataset.md5

    if not args.command or args.command == "list":
        if not args.command or args.command == "list":
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
        root_dir = os.path.dirname(os.path.realpath(dataset_dir))
        tarfile_name = f"{dataset_dir}.tar"
        download_and_extract(resource[args.name], tarfile_name, root_dir, md5.get(args.name))

        junk_files = pathlib.Path(dataset_dir).rglob("._*")
        for j in junk_files:
            os.remove(j)
        os.unlink(tarfile_name)


def run_samples(args):
    print("NOTE:: Will be available in release version to download sample apps from github")
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "sample-apps")
    resource = {
        os.path.basename(f.path): os.path.realpath(f.path)
        for f in os.scandir(samples_dir)
        if f.is_dir()
        if os.path.exists(os.path.join(f.path, "main.py"))
    }

    if not args.command or args.command == "list":
        print("Available Samples are:")
        print("----------------------------------------------------")
        for k, v in resource.items():
            print("  {:<30}: {}".format(k, v))
    else:
        print("Not supported yet!")
        exit(-1)


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
            "main:app",
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
