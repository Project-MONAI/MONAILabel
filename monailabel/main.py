# Copyright (c) MONAI Consortium
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
import logging
import os
import pathlib
import platform
import shutil
import sys

import uvicorn

from monailabel import print_config
from monailabel.config import settings
from monailabel.utils.others.generic import init_log_config

logger = logging.getLogger(__name__)


class Main:
    def __init__(self, loglevel=logging.INFO, actions=("start_server", "apps", "datasets", "plugins")):
        self.actions = set([actions] if isinstance(actions, str) else actions)
        logging.basicConfig(
            level=loglevel,
            format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        )

    def args_start_server(self, parser):
        parser.add_argument("-a", "--app", help="App Directory")
        parser.add_argument("-s", "--studies", help="Studies Directory")
        parser.add_argument(
            "-v", "--verbose", default="INFO", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log Level"
        )

        # --conf key1 value1 --conf key2 value2
        parser.add_argument(
            "-c",
            "--conf",
            nargs=2,
            action="append",
            help="config for the app.  Example: --conf key1 value1 --conf key2 value2",
        )

        parser.add_argument("-i", "--host", default="0.0.0.0", type=str, help="Server IP")
        parser.add_argument("-p", "--port", default=8000, type=int, help="Server Port")

        parser.add_argument("--uvicorn_app", default="monailabel.app:app", type=str, help="Uvicorn App (<path>:<app>)")
        parser.add_argument("--ssl_keyfile", default=None, type=str, help="SSL key file")
        parser.add_argument("--ssl_certfile", default=None, type=str, help="SSL certificate file")
        parser.add_argument("--ssl_keyfile_password", default=None, type=str, help="SSL key file password")
        parser.add_argument("--ssl_ca_certs", default=None, type=str, help="CA certificates file")
        parser.add_argument("--workers", default=None, type=int, help="Number of worker processes")
        parser.add_argument("--limit_concurrency", default=None, type=int, help="Max concurrent connections")
        parser.add_argument("--access_log", action="store_true", help="Enable access log")

        parser.add_argument("-l", "--log_config", default=None, type=str, help="Logging config")
        parser.add_argument("--dryrun", action="store_true", help="Dry run without starting server")

    def args_apps(self, parser):
        parser.add_argument("-d", "--download", action="store_true", help="download app")
        parser.add_argument("-n", "--name", help="Name of the sample app to download", default=None)
        parser.add_argument("-o", "--output", help="Output path to save the app", default=None)
        parser.add_argument("--prefix", default=None)

    def args_datasets(self, parser):
        parser.add_argument("-d", "--download", action="store_true", help="download dataset")
        parser.add_argument("-n", "--name", help="Name of the dataset to download", default=None)
        parser.add_argument("-o", "--output", help="Output path to save the dataset", default=None)
        parser.add_argument("--prefix", default=None)

    def args_plugins(self, parser):
        parser.add_argument("-d", "--download", action="store_true", help="download plugin")
        parser.add_argument("-n", "--name", help="Name of the plugin to download", default=None)
        parser.add_argument("-o", "--output", help="Output path to save the plugin", default=None)
        parser.add_argument("--prefix", default=None)

    def args_parser(self, name="monailabel"):
        parser = argparse.ArgumentParser(name)
        parser.add_argument("-v", "--version", action="store_true", help="print version")

        subparsers = parser.add_subparsers(help="sub-command help")
        if "start_server" in self.actions:
            parser_a = subparsers.add_parser("start_server", help="Start Application Server")
            self.args_start_server(parser_a)
            parser_a.set_defaults(action="start_server")

        if "apps" in self.actions:
            parser_b = subparsers.add_parser("apps", help="list or download sample apps")
            self.args_apps(parser_b)
            parser_b.set_defaults(action="apps")

        if "datasets" in self.actions:
            parser_c = subparsers.add_parser("datasets", help="list or download sample datasets")
            self.args_datasets(parser_c)
            parser_c.set_defaults(action="datasets")

        if "plugins" in self.actions:
            parser_d = subparsers.add_parser("plugins", help="list or download viewer plugins")
            self.args_plugins(parser_d)
            parser_d.set_defaults(action="plugins")

        return parser

    def run(self):
        parser = self.args_parser()
        args = parser.parse_args()

        if args.version:
            print_config()
            exit(0)

        if not hasattr(args, "action"):
            parser.print_usage()
            exit(-1)

        if args.action == "apps":
            self.action_apps(args)
        elif args.action == "datasets":
            self.action_datasets(args)
        elif args.action == "plugins":
            self.action_plugins(args)
        else:
            self.action_start_server(args)

    def action_apps(self, args):
        self._action_xyz(args, "sample-apps", "App", None, shutil.ignore_patterns("logs", "model", "__pycache__"))

    def action_plugins(self, args):
        self._action_xyz(args, "plugins", "Plugin", None, shutil.ignore_patterns("__pycache__"))

    def action_datasets(self, args):
        from monai.apps.datasets import DecathlonDataset
        from monai.apps.utils import download_and_extract

        resource = DecathlonDataset.resource
        md5 = DecathlonDataset.md5

        if not args.download:
            print("Available Datasets are:")
            print("----------------------------------------------------")
            for k, v in resource.items():
                print(f"  {k:<30}: {v}")
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

    def _get_installed_dir(self, prefix, name):
        project_root_absolute = pathlib.Path(__file__).parent.parent.resolve()
        d = os.path.join(project_root_absolute, name)
        if not os.path.exists(d):
            if prefix:
                d = os.path.join(prefix, "monailabel", name)
            else:
                d = os.path.join(sys.prefix, "monailabel", name)
                if not os.path.exists(d):
                    d = os.path.join(pathlib.Path.home(), ".local", "monailabel", name)
        return d

    def _action_xyz(self, args, name, title, exclude, ignore):
        xyz_dir = self._get_installed_dir(args.prefix, name)
        exclude = [exclude] if isinstance(exclude, str) else exclude

        xyz = os.listdir(xyz_dir)
        xyz = [os.path.basename(a) for a in xyz if os.path.isdir(os.path.join(xyz_dir, a))]
        xyz = [p for p in xyz if p not in exclude] if exclude else xyz
        xyz.sort()

        resource = {p: f"{xyz_dir}/{p}" for p in xyz}

        if not args.download:
            print(f"Available {title}s are:")
            print("----------------------------------------------------")
            for k, v in resource.items():
                print(f"  {k:<30}: {v}")
            print("")
        else:
            xyz_dir = os.path.join(xyz_dir, args.name)
            if args.name not in xyz or not os.path.exists(xyz_dir):
                print(f"{title} {args.name} => {xyz_dir} not exists")
                exit(-1)

            output_dir = os.path.realpath(os.path.join(args.output, args.name) if args.output else args.name)
            if os.path.exists(output_dir):
                print(f"Directory already exists: {output_dir}")
                exit(-1)

            if os.path.dirname(output_dir):
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            shutil.copytree(xyz_dir, output_dir, ignore=ignore)
            print(f"{args.name} is copied at: {output_dir}")

    def action_start_server(self, args):
        self.start_server_validate_args(args)
        self.start_server_init_settings(args)

        log_config = init_log_config(args.log_config, args.app, "app.log", args.verbose)

        if args.dryrun:
            return

        uvicorn.run(
            args.uvicorn_app,
            host=args.host,
            port=args.port,
            log_level="info",
            log_config=log_config,
            use_colors=True,
            access_log=args.access_log,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_ca_certs=args.ssl_ca_certs,
            workers=args.workers,
            limit_concurrency=args.limit_concurrency,
        )

    def start_server_validate_args(self, args):
        if not args.app:
            print("APP Directory NOT provided")
            exit(1)

        if not args.studies:
            print("STUDIES Path/Directory NOT provided")
            exit(1)

        if not os.path.exists(args.app):
            print(f"APP Directory {args.app} NOT Found")
            exit(1)

        if (
            not args.studies.startswith("http://")
            and not args.studies.startswith("https://")
            and not os.path.exists(args.studies)
        ):
            print(f"STUDIES Directory {args.studies} NOT Found;  Creating an EMPTY folder/placeholder")
            os.makedirs(args.studies, exist_ok=True)

        args.app = os.path.realpath(args.app)
        if not args.studies.startswith("http://") and not args.studies.startswith("https://"):
            args.studies = os.path.realpath(args.studies)

        for arg in vars(args):
            logger.info(f"USING:: {arg} = {getattr(args, arg)}")

        for k, v in settings.dict().items():
            v = f"'{json.dumps(v)}'" if isinstance(v, list) or isinstance(v, dict) else v
            logger.info(f"ENV SETTINGS:: {k} = {'*' * len(v) if k == 'MONAI_LABEL_DICOMWEB_PASSWORD' else v}")
        logger.info("")

    def start_server_init_settings(self, args):
        # namespace('conf': [['key1','value1'],['key2','value2']])
        conf = {c[0]: c[1] for c in args.conf} if args.conf else {}

        settings.MONAI_LABEL_SERVER_PORT = args.port
        settings.MONAI_LABEL_APP_DIR = args.app
        settings.MONAI_LABEL_STUDIES = args.studies
        settings.MONAI_LABEL_APP_CONF = conf

        dirs = ["model", "lib", "logs", "bin"]
        for d in dirs:
            d = os.path.join(args.app, d)
            if not os.path.exists(d):
                os.makedirs(d)

        sys.path.append(args.app)
        sys.path.append(os.path.join(args.app, "lib"))
        os.environ["PATH"] += os.pathsep + os.path.join(args.app, "bin")

        if args.dryrun:
            export_key = "set " if any(platform.win32_ver()) else "export "
            with open("env.bat" if any(platform.win32_ver()) else ".env", "w") as f:
                for k, v in settings.dict().items():
                    v = f"'{json.dumps(v)}'" if isinstance(v, list) or isinstance(v, dict) else v
                    e = f"{export_key}{k}={v}"
                    f.write(e)
                    f.write(os.linesep)
                    logger.info(e)

                py_path = [os.environ.get("PYTHONPATH", "").rstrip(os.pathsep), args.app, os.path.join(args.app, "lib")]
                py_path = [p for p in py_path if p]
                others = [
                    f"{export_key}PYTHONPATH={os.pathsep.join(py_path)}",
                    f"{export_key}PATH={os.environ['PATH']}",
                ]
                for o in others:
                    f.write(o)
                    f.write(os.linesep)
                    logger.info(o)
        else:
            logger.debug("")
            logger.debug("**********************************************************")
            logger.debug("                  ENV VARIABLES/SETTINGS                  ")
            logger.debug("**********************************************************")
            for k, v in settings.dict().items():
                v = json.dumps(v) if isinstance(v, list) or isinstance(v, dict) else str(v)
                logger.debug(f"{'set' if any(platform.win32_ver()) else 'export'} {k}={v}")
                os.environ[k] = v
            logger.debug("**********************************************************")
            logger.debug("")


if __name__ == "__main__":
    Main().run()
