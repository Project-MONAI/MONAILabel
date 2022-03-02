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

import json
import logging
import os
import tempfile
import time

import large_image
import numpy as np
from client import MONAILabelClient
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser
from histomicstk.utils import compute_tile_foreground_fraction

logging.basicConfig(level=logging.INFO)


def run_monailabel_task(slide_path, level, location, size, tile_position, args, it_kwargs):
    logging.info(f"Run MONAILabel Task... and collect the annotations: {location} => {size}")

    client = MONAILabelClient(server_url=args.server)
    image = os.path.basename(slide_path).replace(".svs", "").replace(".tif", "")
    tile_size = [int(args.analysis_tile_size), int(args.analysis_tile_size)]
    min_poly_area = args.min_poly_area
    output = "dsa"
    params = {
        "level": level,
        "location": location,
        "size": size,
        "tile_size": tile_size,
        "min_poly_area": min_poly_area,
        "output": output,
        "logging": args.loglevel,
    }

    if args.send_image:
        ts = large_image.getTileSource(slide_path)
        tile_info = ts.getSingleTile(
            tile_position=tile_position, format=large_image.tilesource.TILE_FORMAT_NUMPY, **it_kwargs
        )

        # get tile image
        image_np = tile_info["tile"][:, :, :3].astype(np.uint8)
        logging.info(f"Send Image over wire: {image_np.shape}({image_np.dtype})")
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, image_np)
            res_j, _ = client.infer(model=args.model_name, image_in="", file=f.name, params=params)

        with open(res_j, "r") as fp:
            res = json.load(fp)
    else:
        _, res = client.wsi_infer(model=args.model_name, image_in=image, body=params, output=output)
        res = json.loads(res)
    return res["elements"]


def main(args):
    import dask

    total_start_time = time.time()

    print("\n>> CLI Parameters ...\n")
    for arg in vars(args):
        print("USING:: {} = {}".format(arg, getattr(args, arg)))

    if not os.path.isfile(args.inputImageFile):
        raise OSError("Input image file does not exist.")

    if len(args.analysis_roi) != 4:
        raise ValueError("Analysis ROI must be a vector of 4 elements.")

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    print("\n>> Creating Dask client ...\n")
    start_time = time.time()

    c = cli_utils.create_dask_client(args)
    print(c)

    dask_setup_time = time.time() - start_time
    print("Dask setup time = {}".format(cli_utils.disp_time_hms(dask_setup_time)))

    print("\n>> Reading input image ... \n")
    ts = large_image.getTileSource(args.inputImageFile)
    ts_metadata = ts.getMetadata()
    print(json.dumps(ts_metadata, indent=2))

    it_kwargs = {
        "tile_size": {"width": args.analysis_tile_size},
        "level": 0,
    }
    if not process_whole_image:
        it_kwargs["region"] = {
            "left": args.analysis_roi[0],
            "top": args.analysis_roi[1],
            "width": args.analysis_roi[2],
            "height": args.analysis_roi[3],
            "units": "base_pixels",
        }

    tile_fgnd_frac_list = [1.0]
    is_wsi = ts_metadata["magnification"] is not None
    if is_wsi:
        num_tiles = ts.getSingleTile(**it_kwargs)["iterator_range"]["position"]
        print(f"Number of tiles = {num_tiles}")

        if process_whole_image:
            print("\n>> Computing tissue/foreground mask at low-res ...\n")
            start_time = time.time()

            im_fgnd_mask_lres, fgnd_seg_scale = cli_utils.segment_wsi_foreground_at_low_res(ts)
            fgnd_time = time.time() - start_time
            print("low-res foreground mask computation time = {}".format(cli_utils.disp_time_hms(fgnd_time)))

            print("\n>> Computing foreground fraction of all tiles ...\n")
            start_time = time.time()

            tile_fgnd_frac_list = compute_tile_foreground_fraction(
                args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale, it_kwargs
            )
        else:
            tile_fgnd_frac_list = np.full(num_tiles, 1.0)

        num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list >= args.min_fgnd_frac)
        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles
        fgnd_frac_comp_time = time.time() - start_time

        print("Number of foreground tiles = {:d} ({:2f}%%)".format(num_fgnd_tiles, percent_fgnd_tiles))
        print("Tile foreground fraction computation time = {}".format(cli_utils.disp_time_hms(fgnd_frac_comp_time)))

    print("\n>> Running MONAI...\n")
    start_time = time.time()

    tile_annotation_list = []
    skip_count = 0
    for tile in ts.tileIterator(**it_kwargs):
        tile_position = tile["tile_position"]["position"]
        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            print(f"{tile_position} => Skip Tile {tile_fgnd_frac_list[tile_position]} <= {args.min_fgnd_frac}")
            skip_count += 1
            continue

        print(f"{tile_position} => {tile}")
        location = [int(tile["x"]), int(tile["y"])]
        size = [int(tile["width"]), int(tile["height"])]

        cur_annotation_list = dask.delayed(run_monailabel_task)(
            args.inputImageFile,
            level=0,
            location=location,
            size=size,
            tile_position=tile_position,
            args=args,
            it_kwargs=it_kwargs,
        )

        # append result to list
        tile_annotation_list.append(cur_annotation_list)

    print(f"Total Tasks skipped: {skip_count}")
    print(f"Total Tasks To Compute: {len(tile_annotation_list)}")
    tile_annotation_list = dask.delayed(tile_annotation_list).compute()

    annotation_list = [anot for anot_list in tile_annotation_list for anot in anot_list]
    annotation_detection_time = time.time() - start_time

    print("Number of annotations = {}".format(len(annotation_list)))
    print("Total Job time = {}".format(cli_utils.disp_time_hms(annotation_detection_time)))

    print("\n>> Writing annotation file ...\n")
    annotation = {
        "name": f"monailabel-{args.model_name} - " + "wsi" if is_wsi else f"roi{args.analysis_roi}",
        "elements": annotation_list,
    }

    with open(args.outputAnnotationFile, "w") as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)

    total_time_taken = time.time() - total_start_time
    print("Total analysis time = {}".format(cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())

    # (cython) python setup.py build_ext --inplace
    # from argparse import Namespace
    #
    # root = "/localhome/sachi/Data/Pathology
    # args = {
    #     "inputImageFile": f"{root}/Test/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs",
    #     "outputAnnotationFile": f"{root}/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.anot",
    #     "min_poly_area": 40.0,
    #     "analysis_mag": 40.0,
    #     "analysis_roi": [7063.0, 15344.0, 1882.0, 1362.0],
    #     "analysis_tile_size": 2048.0,
    #     "min_fgnd_frac": 0.25,
    #     "model_name": "deepedit",
    #     "num_threads_per_worker": 1,
    #     "num_workers": 1,
    #     "scheduler": "",
    #     "server": "http://10.117.18.128:8000/",
    #     "tile_grouping": 256,
    #     "send_image": False,
    # }
    # ns = Namespace(**args)
    # main(ns)
