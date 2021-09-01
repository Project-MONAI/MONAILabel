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
import os
import glob
from time import sleep

from lib import MyInfer, MyTrain

# from lib.infer_tta import MyInferTTA
from lib.activelearning import TTA, MyStrategy, Epistemic
# from monai.apps import load_from_mmar

# from monailabel.endpoints.utils import BackgroundTask
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.scoring.tta_scoring import TTAScoring
from monailabel.utils.scoring.epistemic_scoring import EpistemicScoring
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monai.networks.nets import UNet
from monai.networks.layers import Norm

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):

        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        # self.mmar = "clara_pt_spleen_ct_segmentation_1"

        self.network = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2 # TODO How do we make this as a hyper-parameter in future?
        )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen",
            description="Active Learning solution to label Spleen Organ over 3D CT Images",
            version=2,
        )

        # TO DO - start scoring after starting App if trained model exists
        # BackgroundTask.run("scoring", request={"method": "TTA"}, params={}, force_sync=False)

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, self.network),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        # infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    # def init_infers(self):
    #     infers = {
    #         "segmentation_spleen": MyInferTTA(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
    #     }
    #     return infers

    def init_trainers(self):
        config = {
            "name": "model_01",
            "pretrained": False,
            "device": "cuda",
            "max_epochs": 1,
            "val_split": 0.2,
            "train_batch_size": 1,
            "val_batch_size": 1,
        }
        return {
            "segmentation_spleen": MyTrain(
                                    self.model_dir,
                                    # load_from_mmar(self.mmar, self.model_dir, pretrained=False),
                                    self.network,
                                    publish_path=self.final_model,
                                    config=config,
            )
        }

    def init_strategies(self):
        return {
            "Epistemic": Epistemic(),
            "TTA": TTA(),
            "random": Random(),
            "first": MyStrategy(),
        }

    def init_scoring_methods(self):
        return {
            "Epistemic": EpistemicScoring(model=self.init_infers()["segmentation_spleen"], plot=False),
        }

    # def create_augmented_imgs(self, request):
    #     # ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    #     for i in ["first", "second", "third"]:
    #         request["label_tag"] = i
    #         self.batch_infer(request)


def main():
    '''

    Workflow to compare TTA against random

    1. Start training with 2 images
    2. Perform TTA
    3. Fetch image and retrain model
    4. Perform TTA
    5. Fetch image and retrain model

    Questions:
    - How to split val and train images
      In train class! Use the method "partition_datalist" to do the partition


    - Why it is not working the epochs specification? I put 100 and it shows 50.
      In init_trainers method using config argument. BUT WHAT IS THE DIFFERENCE BETWEEN THAT AND THE EPOCHS IN REQUEST?

    - How to specify I don't want to use pretrained model to start training?
      In init_trainers method. "load_from_mmar" is the method where the network is being specified


    - A mix of object instantiation and API calls is not possible because
    for API calls we'll need an IP to make the calls and
    object instantiation doesn't have the option to stop the training, it executes line by line

    - The disadvantage of using requests is that I need to first start the App via bash. Not everything is via PyCharm

    '''

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_dir_path = os.path.normpath("/home/vishwesh/Code/MONAILabel/sample-apps/segmentation_spleen_dropout")
    studies_path = os.path.normpath("/home/vishwesh/experiments/monai_label_spleen/small_data")

    al_app = MyApp(app_dir=app_dir_path, studies=studies_path)

    request = {}
    request["device"] = "cuda"
    # request["model"] = "segmentation_spleen"
    request["images"] = "unlabeled"
    # request["save_label"] = True

    # Perform batch inference using augmented images
    al_app.init_infers()
    # al_app.create_augmented_imgs(request)

    request["val_batch_size"] = 1
    # request["epochs"] = 10

    #request["strategy"] = "TTA"
    #request["method"] = "TTA"
    request["strategy"] = "Epistemic"
    request["method"] = "Epistemic"

    # Start training
    al_app.train(request)

    # Perform scoring
    al_app.scoring(request)

    # Fetch next sample. DOES THIS WORK??
    al_app.next_sample(request=request)

    return None


if __name__ == "__main__":
    main()
