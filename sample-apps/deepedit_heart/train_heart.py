import argparse
import distutils.util
import json
import logging
import os
import sys
import time
from typing import Dict
import numpy as np

import torch

from monai.transforms.transform import Transform
from monai.apps.deepgrow.transforms import (
    AddInitialSeedPointd,
    FindDiscrepancyRegionsd,
    AddRandomGuidanced,
    AddGuidanceSignald,
    FindAllValidSlicesd,
)
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.engines import SupervisedTrainer
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    LrScheduleHandler,
    CheckpointSaver,
    MeanDice,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet, UNet
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    AdjustContrastd,
    NormalizeIntensityd,
    ToTensord,
    ToNumpyd,
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    Orientationd,
    RandZoomd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    Spacingd,
    Resized,
)
from monai.utils import set_determinism
from monai.apps.deepgrow.interaction import Interaction

# # To check inner generated images
# from handler import InnerIterSaver
# from inner_event.interaction_deep_edit import Interaction
# from inner_event.Events_deep_edit import DeepEditEvents


class DiscardAddGuidanced(Transform):
    """
    Discard positive and negative points randomly or Add the two channels for inference time
    """

    def __init__(self, image: str = "image", batched: bool = False, ):
        self.image = image
        # What batched means/implies? I see that the dictionary is in the list form instead of numpy array
        self.batched = batched

    def __call__(self, data):
        d: Dict = dict(data)
        # Batched is True
        if isinstance(d[self.image], list):
            image = d[self.image][0]
            # For pure inference time - There is no positive neither negative points
            if len(image.shape) == 3:
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
                d[self.image][0][1] = signal
                d[self.image][0][2] = signal
            else:
                if np.random.choice([False, True], p=[1. / 3, 2. / 3]):
                    print('Deleting P and N points')
                    signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
                    d[self.image][0][1] = signal
                    d[self.image][0][2] = signal
                else:
                    print('Keeping P and N points')
        # Batched is False
        else:
            image = d[self.image]
            # For pure inference time - There is no positive neither negative points
            if len(image.shape) == 3:
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
                d[self.image][1] = signal
                d[self.image][2] = signal
            else:
                if np.random.choice([False, True], p=[1. / 3, 2. / 3]):
                    print('Deleting P and N points')
                    signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
                    d[self.image][1] = signal
                    d[self.image][2] = signal
                else:
                    print('Keeping P and N points')

        return d


def get_network(network, channels, dimensions):
    if network == 'unet':
        if channels == 16:
            features = (16, 32, 64, 128, 256)
        elif channels == 32:
            features = (32, 64, 128, 256, 512)
        else:
            features = (64, 128, 256, 512, 1024)
        logging.info('Using Unet with features: {}'.format(features))
        network = UNet(dimensions=dimensions, in_channels=3, out_channels=1, channels=features, strides=[2, 2, 2, 2],
                       norm=Norm.BATCH)
    elif network == 'bunet':
        if channels == 16:
            features = (16, 32, 64, 128, 256, 16)
        elif channels == 32:
            features = (32, 64, 128, 256, 512, 32)
        else:
            features = (64, 128, 256, 512, 1024, 64)
        logging.info('Using BasicUnet with features: {}'.format(features))
        network = BasicUNet(dimensions=dimensions, in_channels=3, out_channels=1, features=features)

    elif network == 'dynunet':
        logging.info('Using dynunet')
        network = DynUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True)
    return network


def get_pre_transforms(roi_size, model_size):
    t = [
        LoadImaged(keys=('image', 'label')),
        RandZoomd(keys=('image', 'label'), prob=0.4, min_zoom=0.3, max_zoom=1.9, mode=("bilinear", "nearest")),
        AddChanneld(keys=('image', 'label')),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys='image'),
        # Using threshold to crop images
        CropForegroundd(keys=('image', 'label'), source_key='image', select_fn=lambda x: x > 2.9, margin=3),
        RandAdjustContrastd(keys='image', gamma=6),
        RandHistogramShiftd(keys='image', num_control_points=8, prob=0.5),
        Resized(keys=('image', 'label'), spatial_size=model_size, mode=('area', 'nearest')),
        FindAllValidSlicesd(label='label', sids='sids'),
        AddInitialSeedPointd(label='label', guidance='guidance', sids='sids'),
        AddGuidanceSignald(image='image', guidance='guidance'),
        DiscardAddGuidanced(image='image'),
        ToTensord(keys=('image', 'label'))
    ]
    return Compose(t)


def get_click_transforms():
    return Compose([
        Activationsd(keys='pred', sigmoid=True),
        ToNumpyd(keys=('image', 'label', 'pred', 'probability', 'guidance')),
        FindDiscrepancyRegionsd(label='label', pred='pred', discrepancy='discrepancy', batched=True),
        AddRandomGuidanced(guidance='guidance', discrepancy='discrepancy', probability='probability', batched=True),
        # apply gaussian to guidance and add them as new channel
        AddGuidanceSignald(image='image', guidance='guidance', batched=True),
        DiscardAddGuidanced(image='image', batched=True),
        ToTensord(keys=('image', 'label'))
    ])


def get_post_transforms():
    return Compose([
        Activationsd(keys='pred', sigmoid=True),
        AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.5)
    ])


def get_loader(args, pre_transforms):
    # Reading images
    data_dir = os.path.join(args.input)
    with open(os.path.join(data_dir, 'dataset.json')) as file:
        data = json.load(file)
    datalist = [{"image": os.path.join(args.input, d['image']),
                 "label": os.path.join(args.input, d['labels'][0])} for d in data['objects']]
    total_d = len(datalist)
    total_l = len(datalist)

    # Creating dataloader
    train_ds = PersistentDataset(datalist, pre_transforms, cache_dir=args.cache_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0)
    logging.info('Total Records used for Training is: {}/{}/{}'.format(len(train_ds), total_l, total_d))

    return train_loader


def create_trainer(args):
    set_determinism(seed=args.seed)

    device = torch.device("cuda" if args.use_gpu else "cpu")

    pre_transforms = get_pre_transforms(args.roi_size, args.model_size)
    click_transforms = get_click_transforms()
    post_transform = get_post_transforms()

    train_loader = get_loader(args, pre_transforms)

    # define training components
    network = get_network(args.network, args.channels, args.dimensions).to(device)

    if args.resume:
        logging.info('Loading Network...')
        map_location = {"cuda:0": "cuda:0"}
        network.load_state_dict(torch.load(args.model_filepath, map_location=map_location))

    loss_function = DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(log_dir=args.output, tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaver(save_dir=args.output, save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
                        save_interval=args.save_interval * 2, save_final=True, final_filename='checkpoint.pt'),
        # # To check inner generated images
        # InnerIterSaver(),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            transforms=click_transforms,
            max_interactions=args.max_train_interactions,
            key_probability='probability',
            train=True),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        post_transform=post_transform,
        amp=args.amp,
        key_train_metric={
            "train_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"])
            )
        },
        train_handlers=train_handlers,
        # # To check inner generated images
        # event_names=[DeepEditEvents],
    )
    return trainer


def run(args):
    args.roi_size = json.loads(args.roi_size)
    args.model_size = json.loads(args.model_size)

    # Printing arguments
    for arg in vars(args):
        logging.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    if not os.path.exists(args.output):
        logging.info('output path [{}] does not exist. creating it now.'.format(args.output))
        os.makedirs(args.output, exist_ok=True)

    trainer = create_trainer(args)

    # Running training
    start_time = time.time()
    trainer.run()
    end_time = time.time()

    # Saving model pt
    logging.info('Total Training Time {}'.format(end_time - start_time))
    logging.info('Saving Final PT Model')
    torch.save(trainer.network.state_dict(), os.path.join(args.output, 'model-final.pt'))

    # Saving TorchScript model ts
    logging.info('Saving TorchScript Model')
    model_ts = torch.jit.script(trainer.network)
    torch.jit.save(model_ts, os.path.join(args.output, 'model-final.ts'))


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--dimensions', type=int, default=3)

    parser.add_argument('-n', '--network', default='dynunet', choices=['unet', 'bunet', 'dynunet'])
    parser.add_argument('-c', '--channels', type=int, default=32)
    parser.add_argument('-i', '--input',
                        default='/home/adp20local/Documents/ActiveLearningTechniques/deepedit_simplified/datastore/heart/')
    parser.add_argument('-o', '--output', default='model')

    parser.add_argument('-g', '--use_gpu', type=strtobool, default='true')
    parser.add_argument('-a', '--amp', type=strtobool, default='false')

    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default=None)

    parser.add_argument('-r', '--resume', type=strtobool, default='false')
    parser.add_argument('--roi_size', default="[128, 128, 128]")
    parser.add_argument('--model_size', default="[128, 128, 128]")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-it', '--max_train_interactions', type=int, default=20)

    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
