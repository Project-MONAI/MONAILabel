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
import distutils.util
import json
import logging
import os
import shutil
import sys

import yaml

from monailabel.interfaces.utils.app import app_instance

logger = logging.getLogger(__name__)


def test_infer(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    logger.info("Running Inference Task: {}".format(args.model))

    response = None
    for _ in range(args.runs):
        request = {
            "model": args.model,
            "image": args.input,
            "params": json.loads(args.params),
            "device": args.device,
        }
        response = app.infer(request=request)

    res_img = response.get("label")
    res_json = response.get("params")
    if res_img:
        result_image = args.output
        print(f"Move: {res_img} => {result_image}")
        shutil.move(res_img, result_image)
        os.chmod(result_image, 0o777)
        print("Check Result file: {}".format(result_image))

    print("Result JSON: {}".format(res_json))


def test_train(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    logger.info("Running Training Task: {}".format(args.name))

    request = {
        "name": args.name,
        "device": args.device,
        "epochs": args.epochs,
        "amp": args.amp,
    }
    app.train(request)


def test_info(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    info = app.info()

    class MyDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow, False)

    yaml.dump(
        info,
        sys.stdout,
        Dumper=MyDumper,
        sort_keys=False,
        default_flow_style=False,
        width=120,
        indent=2,
    )


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-a", "--app", required=True)
    parser.add_argument("-s", "--studies", required=True)
    parser.add_argument("--device", default="cuda")

    subparsers = parser.add_subparsers(help="sub-command help")

    parser_a = subparsers.add_parser("infer", help="infer help")
    parser_a.add_argument("-m", "--model", required=True, help="Pre-Trained Model for inference")
    parser_a.add_argument("-i", "--input", required=True, help="Input Image file")
    parser_a.add_argument("-o", "--output", required=True, help="Output Label file")
    parser_a.add_argument("-p", "--params", default="{}", help="Input Params for inference")
    parser_a.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of times to run same inference",
    )
    parser_a.set_defaults(test="infer")

    parser_b = subparsers.add_parser("train", help="train help")
    parser_b.add_argument("-n", "--name", required=True, help="Name of Train task/Output folder name")
    parser_b.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser_b.add_argument("--amp", type=strtobool, default="true", help="Use AMP")
    parser_b.set_defaults(test="train")

    parser_c = subparsers.add_parser("info", help="info help")
    parser_c.set_defaults(test="info")

    args = parser.parse_args()
    if not hasattr(args, "test"):
        parser.print_usage()
        exit(-1)

    args.app = os.path.realpath(args.app)
    args.studies = os.path.realpath(args.studies)
    for arg in vars(args):
        print("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.test == "infer":
        test_infer(args)
    elif args.test == "train":
        test_train(args)
    elif args.test == "info":
        test_info(args)
    else:
        parser.print_help()
