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

import glob
import logging
import os
from typing import Dict

from lib import MyInfer, MyTrain
from monai.networks.nets import UNet
from PIL import Image

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.network = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Segmentation - Pathology",
            description="Active Learning solution for Pathology",
        )

    def init_infers(self) -> Dict[str, InferTask]:
        return {
            "segmentation": MyInfer([self.pretrained_model, self.final_model], self.network),
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        return {
            "segmentation": MyTrain(
                model_dir=self.model_dir,
                network=self.network,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"max_epochs": 10, "train_batch_size": 1},
            )
        }


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse

    from monailabel.config import settings

    settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False
    settings.MONAI_LABEL_DATASTORE_FILE_EXT = ["*.png"]
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/Camelyon/monai_dataset")
    # parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/Camelyon/dataset/training/images")
    parser.add_argument("-s", "--studies", default="/local/sachi/Data/Pathology/Camelyon/dataset/testing/images")
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-o", "--output", default="model_01")
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "use_pretrained_model": "false",
        "auto_update_scoring": "false",
    }

    app = MyApp(app_dir, studies, conf)
    # app.train(
    #     request={
    #         "name": args.output,
    #         "model": "segmentation",
    #         "max_epochs": args.epoch,
    #         "dataset": "Dataset",
    #         "train_batch_size": 16,
    #         "val_batch_size": 4,
    #         "multi_gpu": True,
    #     }
    # )

    res = app.infer({"model": "segmentation", "image": "test_001_Annotation 0_0x0", "result_extension": ".nii"})
    print(res)


def prepare_dataset():
    root_dir = "/local/sachi/Data/Pathology/Camelyon"
    images = sorted(glob.glob(f"{root_dir}/dataset/training/images/*.png"))
    labels = sorted(glob.glob(f"{root_dir}/dataset/training/labels/*.png"))

    ds = {}
    img_size = 4096
    patch_size = 64

    patch_per_side = img_size // patch_size
    grid_size = patch_per_side * patch_per_side

    print(f"patch per row/side = {patch_per_side}")

    img_flat = []
    lab_flat = []

    for image, label in zip(images, labels):
        assert os.path.basename(image) == os.path.basename(label)
        i = None
        l = None

        idx = 0
        for x_idx in range(patch_per_side):
            for y_idx in range(patch_per_side):
                x_start = x_idx * patch_size
                x_end = x_start + patch_size

                y_start = y_idx * patch_size
                y_end = y_start + patch_size

                ip = i[x_start:x_end, y_start:y_end, :]
                lp = l[x_start:x_end, y_start:y_end]
                # print(f"{idx} => {ip.shape} => {lp.shape} => x[{x_start}:{x_end}] => y[{y_start}:{y_end}] => sum: {np.sum(ip)} -> {np.sum(lp)}")

                img_flat.append(ip)
                lab_flat.append(lp)
                idx += 1


if __name__ == "__main__":
    main()
