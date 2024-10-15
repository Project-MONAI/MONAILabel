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

import logging
import os
from typing import Dict

from lib.infers.sam2 import Sam2

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - SAM2 ({monailabel.__version__})",
            description="SAM2 model",
            version=monailabel.__version__,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {"sam2": Sam2()}
        return infers


"""
Example to run train/infer/batch infer/scoring task(s) locally without actually running MONAI Label Server

More about the available app methods, please check the interface monailabel/interfaces/app.py

"""


def main():
    import argparse
    from pathlib import Path

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Dataset/SAM2"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {}

    app = MyApp(app_dir, studies, conf)

    # Infer
    image_id = "spleen_10"
    image_path = args.studies + "/" + image_id + ".nii.gz"

    res = app.infer(request={"model": "sam2", "image": image_id, "device": "cuda"})
    # label = res["file"]
    # label_json = res["params"]
    # test_dir = os.path.join(args.studies, "test_labels")
    # os.makedirs(test_dir, exist_ok=True)
    #
    # label_file = os.path.join(test_dir, image_id + file_ext(image_path))
    # shutil.move(label, label_file)
    #
    # print(label_json)
    print(f"++++ Image File: {image_path}")
    print(f"++++ Inference Result: {res}")


if __name__ == "__main__":
    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()
