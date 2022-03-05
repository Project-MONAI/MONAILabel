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

import logging
import time

from cli.client import MONAILabelClient
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.INFO)


def main(args):
    print("\n>> CLI Parameters ...\n")
    for arg in vars(args):
        print("USING:: {} = {}".format(arg, getattr(args, arg)))

    print("\n>> Running MONAI...\n")
    start_time = time.time()
    params = {
        "name": args.train_name,
        "model": args.model,
        "max_epochs": args.max_epochs,
        "dataset": args.dataset,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "multi_gpu": args.multi_gpu,
        "val_split": args.val_split,
    }

    client = MONAILabelClient(server_url=args.server)
    print("Will stop any previous running tasks...")
    client.train_stop()

    print("Trigger new Training job...")
    client.train_start(args.model, params)

    total_time_taken = time.time() - start_time
    print("Training Job Triggered/Started = {}".format(cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
