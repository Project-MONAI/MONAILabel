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
import os
import time
from pathlib import Path

from cli.client import MONAILabelClient

logging.basicConfig(level=logging.INFO)


def main(args):
    total_start_time = time.time()

    print("\n>> CLI Parameters ...\n")
    for arg in vars(args):
        print(f"USING:: {arg} = {getattr(args, arg)}")

    if not os.path.isfile(args.inputImageFile):
        raise OSError("Input image file does not exist.")

    if len(args.analysis_roi) != 4:
        raise ValueError("Analysis ROI must be a vector of 4 elements.")

    location = [args.analysis_roi[0], args.analysis_roi[1]]
    size = [args.analysis_roi[2], args.analysis_roi[3]]

    start_time = time.time()
    logging.info(f"Run MONAILabel Task... and collect the annotations: {location} => {size}")
    logging.info(f"For Server Logs click/open:  {args.server.rstrip('/')}/logs/?refresh=3")

    client = MONAILabelClient(server_url=args.server)
    image = Path(os.path.basename(args.inputImageFile)).stem
    tile_size = [int(args.analysis_tile_size), int(args.analysis_tile_size)]
    min_poly_area = args.min_poly_area
    output = "dsa"
    body = {
        "level": args.analysis_level,
        "location": location,
        "size": size,
        "tile_size": tile_size,
        "min_poly_area": min_poly_area,
        "output": output,
    }
    extra_params = json.loads(args.extra_params)
    body["params"] = {
        "logging": args.loglevel,
        "max_workers": args.max_workers,
        **extra_params,
    }

    _, res = client.wsi_infer(model=args.model_name, image_in=image, body=body, output=output)
    logging.info(f"Annotation Detection Time = {round(time.time() - start_time, 2)}")

    res = json.loads(res)
    annotation_list = res["elements"]
    logging.info(f"Number of annotations = {len(annotation_list)}")

    logging.info("Writing annotation file...")
    annotation = {
        "name": f"monailabel-{args.model_name} - {args.analysis_roi}",
        "elements": annotation_list,
    }

    with open(args.outputAnnotationFile, "w") as annotation_file:
        json.dump(annotation, annotation_file)

    total_time_taken = time.time() - total_start_time
    logging.info(f"Total analysis time = {round(total_time_taken, 2)}")


if __name__ == "__main__":
    from histomicstk.cli.utils import CLIArgumentParser

    main(CLIArgumentParser().parse_args())

    # (cython) python setup.py build_ext --inplace
    # from argparse import Namespace
    #
    # root = "/localhome/sachi/Data/Pathology"
    # args = {
    #     "inputImageFile": f"{root}/Test/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs",
    #     "outputAnnotationFile": f"{root}/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.anot",
    #     "min_poly_area": 40.0,
    #     "analysis_level": 0,
    #     "analysis_roi": [-1, -1, -1, -1],
    #     # "analysis_roi": [7063.0, 15344.0, 1882.0, 1362.0],
    #     "analysis_tile_size": 2048.0,
    #     "model_name": "deepedit_nuclei",
    #     "server": "http://10.117.18.128:8000/",
    #     "extra_params": "{}",
    #     "loglevel": "ERROR",
    #     "max_workers": 8,
    # }
    # ns = Namespace(**args)
    # main(ns)
