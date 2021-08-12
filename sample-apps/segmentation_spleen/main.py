import logging
import os
import json
from lib import MyInfer, MyStrategy, MyTrain
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monai.networks.nets import UNet
from monai.networks.layers import Norm

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(app_dir, studies, os.path.join(self.model_dir, "train_stats.json"))

    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_dropout_debug"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        '''
        if not os.path.exists(load_path) and request.get("pretrained", True):
            load_path = None
            network = load_from_mmar(self.mmar, self.model_dir)
        else:
            network = load_from_mmar(self.mmar, self.model_dir, pretrained=False)
        '''

        network = UNet(
                        dimensions=3,
                        in_channels=1,
                        out_channels=1,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                        norm=Norm.BATCH,
                        dropout=0.2
                    )

        # Datalist for train/validation
        #train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

        # Load Json file
        data_root = self.studies

        json_file_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data/dataset_0.json')
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        json_file.close()

        train_d = json_data['training']
        val_d = json_data['validation']

        # Add data_root to json
        for idx, each_sample in enumerate(train_d):
            train_d[idx]['image'] = os.path.join(data_root, train_d[idx]['image'])
            train_d[idx]['label'] = os.path.join(data_root, train_d[idx]['label'])

        for idx, each_sample in enumerate(val_d):
            val_d[idx]['image'] = os.path.join(data_root, val_d[idx]['image'])
            val_d[idx]['label'] = os.path.join(data_root, val_d[idx]['label'])

        print('Debug here')

        task = MyTrain(
            output_dir=output_dir,
            train_datalist=train_d,
            val_datalist=val_d,
            network=network,
            load_path=load_path,
            publish_path=self.final_model,
            stats_path=self.train_stats_path,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.0001),
            val_split=request.get("val_split", 0.2),
            max_epochs=request.get("epochs", 1),
            amp=request.get("amp", True),
            train_batch_size=request.get("train_batch_size", 2),
            val_batch_size=request.get("val_batch_size", 1),
        )
        return task()

def main():
    app_dir_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen')
    studies_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data')
    al_app = MyApp(app_dir=app_dir_path, studies=studies_path)
    request = {}
    request["val_batch_size"] = 1
    request["epochs"] = 50
    al_app.train(request=request)
    return None
if __name__=="__main__":
    main()