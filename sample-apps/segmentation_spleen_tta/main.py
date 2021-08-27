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

from lib import MyInfer, MyTrain

# from lib.infer_tta import MyInferTTA
from lib.activelearning import TTA, MyStrategy
from monai.apps import load_from_mmar

# from monailabel.endpoints.utils import BackgroundTask
from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.scoring.tta_scoring import TTAScoring

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="Segmentation - Spleen",
            description="Active Learning solution to label Spleen Organ over 3D CT Images",
            version=2,
        )

        # TO DO
        # BackgroundTask.run("scoring", request={"method": "TTA"}, params={}, force_sync=False)

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    # def init_infers(self):
    #     infers = {
    #         "segmentation_spleen": MyInferTTA(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
    #     }
    #     return infers

    def init_trainers(self):
        return {
            "segmentation_spleen": MyTrain(
                self.model_dir, load_from_mmar(self.mmar, self.model_dir), publish_path=self.final_model
            )
        }

    def init_strategies(self):
        return {
            "TTA": TTA(),
            "random": Random(),
            "first": MyStrategy(),
        }

    def init_scoring_methods(self):
        return {
            "TTA": TTAScoring(model=self.init_infers()["segmentation_spleen"]),
        }

    # def create_augmented_imgs(self, request):
    #     # ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    #     for i in ["first", "second", "third"]:
    #         request["label_tag"] = i
    #         self.batch_infer(request)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_dir_path = os.path.normpath("/home/adp20local/Documents/MONAILabel/sample-apps/segmentation_spleen_tta")
    studies_path = os.path.normpath("/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/train_small")
    al_app = MyApp(app_dir=app_dir_path, studies=studies_path)
    request = {}
    request["device"] = "cuda"
    request["model"] = "segmentation_spleen"
    request["images"] = "unlabeled"
    request["save_label"] = True
    # Perform batch inference using augmented images
    al_app.init_infers()
    # al_app.create_augmented_imgs(request)

    request["val_batch_size"] = 1
    request["epochs"] = 1
    request["strategy"] = "TTA"
    request["method"] = "TTA"
    # Perform scoring
    al_app.scoring(request)
    # Fetch next sample
    al_app.next_sample(request=request)
    al_app.next_sample(request=request)
    return None


if __name__ == "__main__":
    main()
