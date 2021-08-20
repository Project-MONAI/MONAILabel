import logging
import os

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monailabel.utils.others.planner import ExperimentPlanner

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.network = DynUNetV1(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")
        self.spatial_size = None
        self.target_spacing = None

        self.download(
            [
                (
                    self.pretrained_model,
                    "https://github.com/Project-MONAI/MONAILabel/releases/download/data/deepedit_spleen.pt",
                ),
            ]
        )

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            name="DeepEdit - Spleen",
            description="Active learning solution using DeepEdit to label spleen over 3D CT Images",
            version=2,
        )

    def experiment_planner(self):
        # Experiment planner
        self.planner = ExperimentPlanner(datastore=self.datastore())
        self.spatial_size = self.planner.get_target_img_size()
        self.target_spacing = self.planner.get_target_spacing()
        print("Available GPU 0 memory: ", str(self.planner.get_gpu_memory_map().values()))

    def init_infers(self):
        self.experiment_planner()
        print(self.target_spacing)
        print(self.spatial_size)
        return {
            "deepedit": Deepgrow(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
            "spleen": Segmentation(
                [self.pretrained_model, self.final_model],
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
            ),
        }

    def init_trainers(self):
        return {
            "deepedit_spleen": MyTrain(
                self.model_dir,
                self.network,
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                load_path=self.pretrained_model,
                publish_path=self.final_model,
                config={"pretrained": False},
            )
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_dir_path = os.path.normpath("/home/adp20local/Documents/MONAILabel/sample-apps/deepedit_spleen")
    studies_path = os.path.normpath("/home/adp20local/Documents/Datasets/monailabel_datasets/spleen/train_small")
    al_app = MyApp(app_dir=app_dir_path, studies=studies_path)
    request = {}
    request["val_batch_size"] = 1
    request["epochs"] = 1
    al_app.train(request=request)
    return None


if __name__ == "__main__":
    main()
