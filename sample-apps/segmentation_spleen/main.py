import logging
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from lib import MyInfer, MyStrategy, MyTrain
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

from monai.data import DataLoader, PersistentDataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(app_dir, studies, os.path.join(self.model_dir, "train_stats.json"))


    '''
    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers
    '''

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_infer_test"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        #load_path = os.path.join(output_dir, "model.pt")
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
                        out_channels=2,
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
            load_path=None,
            publish_path=self.final_model,
            stats_path=self.train_stats_path,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.001),
            val_split=request.get("val_split", 0.2),
            max_epochs=request.get("epochs", 1),
            amp=request.get("amp", False),
            train_batch_size=request.get("train_batch_size", 4),
            val_batch_size=request.get("val_batch_size", 1),
        )

        #self.batch_infer()

        return task()


def val_pre_transforms():
    return [
        LoadImaged(keys=("image", "label")),
        AddChanneld(keys=("image", "label")),
        Spacingd(
            keys=("image", "label"),
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=("image", "label"), source_key="image"),
        ToTensord(keys=("image", "label")),
    ]

def val_data_loader(_val_datalist):
    return (
        DataLoader(
            dataset=PersistentDataset(_val_datalist, val_pre_transforms(), cache_dir=None),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
#        if _val_datalist and len(_val_datalist) > 0
#        else None
    )


def entropy_3d_volume(vol_input):
    # The input is assumed with repetitions, channels and then volumetric data
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    reps = dims[0]
    entropy = np.zeros(dims[2:], dtype='float32')

    # Threshold values less than or equal to zero
    threshold = 0.00005
    vol_input[vol_input <= 0] = threshold

    # Looping across channels as each channel is a class
    if len(dims) == 5:
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy
    else:
        t_vol = np.squeeze(vol_input)
        t_sum = np.sum(t_vol, axis=0)
        t_avg = np.divide(t_sum, reps)
        t_log = np.log(t_avg)
        t_entropy = -np.multiply(t_avg, t_log)
        entropy = entropy + t_entropy

    return entropy

def main():

    # TODO Notes run 4 iterations generate model directory names in the format "model_1", "model_2" ...
    # TODO In the json list get an unlabeled pool of data, start with 2 or 3 volumes, keep 9 for validation
    # TODO Run uncertainty on post procesed activated probability maps
    # TODO Run inference and compute uncertainty for all unlabeled data, get file names attached with it

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_dir_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen')
    studies_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data')
    al_app = MyApp(app_dir=app_dir_path, studies=studies_path)
    request = {}
    request["val_batch_size"] = 1
    request["epochs"] = 1
    #al_app.train(request=request)

    device = torch.device("cuda:0")

    network = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.2
    )

    model_weights_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/model/model_dropout_02_bs4_sameres/model.pt')
    ckpt = torch.load(model_weights_path)
    network.load_state_dict(ckpt)
    network.to(device=device)
    print('Weights Loaded Succesfully')

    #network.eval()
    network.train()
    # TODO Batch Inference
    # Send the datastore
    #al_app.batch_infer(datastore=)
    #infers = MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir))

    data_root = studies_path
    json_file_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data/dataset_0.json')
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    json_file.close()

    train_d = json_data['training']
    val_d = json_data['validation'][0:2]

    # Add data_root to json
    for idx, each_sample in enumerate(train_d):
        train_d[idx]['image'] = os.path.join(data_root, train_d[idx]['image'])
        train_d[idx]['label'] = os.path.join(data_root, train_d[idx]['label'])

    for idx, each_sample in enumerate(val_d):
        val_d[idx]['image'] = os.path.join(data_root, val_d[idx]['image'])
        val_d[idx]['label'] = os.path.join(data_root, val_d[idx]['label'])

    val_loader = val_data_loader(val_d)

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            accum_val_inputs = []

            for mc in range(10):
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, network)

                # Activate the output with Softmax
                val_act_outputs = torch.softmax(val_outputs, dim=1)

                # Accumulate
                accum_val_inputs.append(val_act_outputs)


            # Stack it up
            accum_tensor = torch.stack(accum_val_inputs)

            # Squeeze
            accum_tensor = torch.squeeze(accum_tensor)

            # Send to CPU
            accum_numpy = accum_tensor.to('cpu').numpy()
            #accum_numpy = accum_numpy[:, 1, :, :, :]
            # Generate Entropy Map and Plot all slices
            entropy = entropy_3d_volume(accum_numpy)

            # Plot with matplotlib and save all slices
            plt.imshow(np.squeeze(entropy[:, :, 50]))
            plt.colorbar()
            plt.title('Dropout Uncertainty')
            plt.show()
            print('Debug here')

    return None
if __name__=="__main__":
    main()