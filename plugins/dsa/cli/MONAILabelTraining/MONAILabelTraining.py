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

import json
import logging
import time

from cli.client import MONAILabelClient
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.INFO)


def get_model_names(args):
    client = MONAILabelClient(server_url=args.server)
    for model_name in client.info()["models"]:
        print("<element>%s</element>" % model_name)


def main(args):
    if args.model_name == "__datalist__":
        return get_model_names(args)
    print("\n>> CLI Parameters ...\n")
    for arg in vars(args):
        print(f"USING:: {arg} = {getattr(args, arg)}")

    print("\n>> Running MONAI...\n")
    start_time = time.time()
    params = {
        "name": args.train_name,
        "max_epochs": args.max_epochs,
        "dataset": args.dataset,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "val_split": args.val_split,
        "dataset_limit": args.dataset_limit,
        "dataset_max_region": args.dataset_max_region,
        "dataset_randomize": args.dataset_randomize,
        "girder_api_url": args.girderApiUrl,
        "girder_token": args.girderToken,
    }
    extra_params = json.loads(args.extra_params)
    params.update(extra_params)

    client = MONAILabelClient(server_url=args.server)
    if args.stop_previous:
        print("Will stop any previous running tasks...")
        client.train_stop()

    print("Trigger Training job...")
    client.train_start(model=args.model_name, params=params)

    total_time_taken = time.time() - start_time
    print(f"Training Job Triggered/Started = {cli_utils.disp_time_hms(total_time_taken)}")


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
