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
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.INFO)


def fetch_annotations(args, tiles=None):
    total_start_time = time.time()

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
        "girder_api_url": args.girderApiUrl,
        "girder_token": args.girderToken,
    }
    extra_params = json.loads(args.extra_params)
    body["params"] = {
        "wsi_tiles": tiles,
        **extra_params,
    }

    _, res = client.wsi_infer(model=args.model_name, image_in=image, body=body, output=output)
    logging.info(f"Annotation Detection Time = {round(time.time() - start_time, 2)}")

    res = str(res, encoding="utf-8")
    with open(args.outputAnnotationFile, "w") as fp:
        fp.write(res)

    # print last but one line of result (description)
    try:
        d = json.loads(json.loads(res[-5000:].split("\n")[-2:][0].replace('"description": ', "")))
        logging.info(f"\n{json.dumps(d, indent=2)}")
    except Exception:
        pass

    total_time_taken = time.time() - total_start_time
    logging.info(f"Total Annotation Fetch time = {round(total_time_taken, 2)}")


def get_model_names(args):
    client = MONAILabelClient(server_url=args.server)
    for model_name in client.info()["models"]:
        print("<element>%s</element>" % model_name)


def main(args):
    if args.model_name == "__datalist__":
        return get_model_names(args)

    total_start_time = time.time()
    logging.info("CLI Parameters ...\n")
    for arg in vars(args):
        logging.info(f"USING:: {arg} = {getattr(args, arg)}")

    if not os.path.isfile(args.inputImageFile):
        raise OSError("Input image file does not exist.")

    if len(args.analysis_roi) != 4:
        raise ValueError("Analysis ROI must be a vector of 4 elements.")

    logging.info(">> Reading input image ... \n")
    tiles = []

    if args.min_fgnd_frac >= 0:
        import large_image
        import numpy as np
        from histomicstk.cli import utils as cli_utils
        from histomicstk.utils import compute_tile_foreground_fraction

        ts = large_image.getTileSource(args.inputImageFile)
        ts_metadata = ts.getMetadata()
        logging.info(json.dumps(ts_metadata, indent=2))

        it_kwargs = {
            "tile_size": {"width": args.analysis_tile_size},
            "scale": {"magnification": 0},
        }
        if not np.all(np.array(args.analysis_roi) <= 0):
            it_kwargs["region"] = {
                "left": args.analysis_roi[0],
                "top": args.analysis_roi[1],
                "width": args.analysis_roi[2],
                "height": args.analysis_roi[3],
                "units": "base_pixels",
            }

        num_tiles = ts.getSingleTile(**it_kwargs)["iterator_range"]["position"]
        logging.info(f"Number of tiles = {num_tiles}")

        logging.info(">> Computing tissue/foreground mask at low-res ...\n")
        start_time = time.time()

        im_fgnd_mask_lres, fgnd_seg_scale = cli_utils.segment_wsi_foreground_at_low_res(ts)
        logging.info(f"low-res foreground mask computation time = {round(time.time() - start_time, 2)}")

        logging.info(">> Computing foreground fraction of all tiles ...\n")
        start_time = time.time()

        tile_fgnd_frac_list = compute_tile_foreground_fraction(
            args.inputImageFile,
            im_fgnd_mask_lres,
            fgnd_seg_scale,
            it_kwargs,
        )

        num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list >= args.min_fgnd_frac)
        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        logging.info(f"Number of foreground tiles = {num_fgnd_tiles:d} ({percent_fgnd_tiles:2f}%%)")
        logging.info(f"Tile foreground fraction computation time = {round(time.time() - start_time, 2)}")

        skip_count = 0
        for tile in ts.tileIterator(**it_kwargs):
            tile_position = tile["tile_position"]["position"]
            location = [int(tile["x"]), int(tile["y"])]
            size = [int(tile["width"]), int(tile["height"])]
            frac = tile_fgnd_frac_list[tile_position]

            if frac <= args.min_fgnd_frac:
                # logging.info(f"Skip:: {tile_position} => {location}, {size} ({frac} <= {args.min_fgnd_frac})")
                skip_count += 1
                continue

            # logging.info(f"Add:: {tile_position} => {location}, {size}")
            tiles.append(
                {
                    "location": location,
                    "size": size,
                }
            )

        logging.info(f"Total Tiles skipped: {skip_count}")
        logging.info(f"Total Tiles To Annotate: {len(tiles)}")

    fetch_annotations(args, tiles)
    logging.info(f"Total Job time = {round(time.time() - total_start_time)}")
    print("All done!")


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())

    # (cython) python setup.py build_ext --inplace
    # from argparse import Namespace
    #
    # root = "/localhome/sachi/Datasets"
    # args = {
    #     "inputImageFile": f"{root}/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs",
    #     "outputAnnotationFile": f"{root}/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.anot",
    #     "min_poly_area": 80.0,
    #     "analysis_level": 0,
    #     "analysis_roi": [-1, -1, -1, -1],
    #     "analysis_tile_size": 1024.0,
    #     "min_fgnd_frac": 0.25,
    #     "model_name": "deepedit_nuclei",
    #     "extra_params": "{}",
    #     "num_threads_per_worker": 1,
    #     "num_workers": 0,
    #     "scheduler": "",
    #     "server": "http://10.117.16.216:8000/",
    #     "girderApiUrl": "http://10.117.16.216/api/v1",
    #     "girderToken": "zBsr184BByiRK0BUyMMB01v3O8kTqkXPbqxndpfi",
    # }
    # ns = Namespace(**args)
    # main(ns)
