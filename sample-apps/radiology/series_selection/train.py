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
import argparse
import glob
import logging
import os
import sys

import monai
import numpy as np
import torch
from ignite.engine import Events, _prepare_batch, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from monai.data import DataLoader, decollate_batch
from monai.handlers import ROCAUC, StatsHandler, TensorBoardStatsHandler, stopping_fn_from_metric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, NormalizeIntensityd, RandRotate90d, Resized


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("-v", "--val", type=float, default=0.2)
    parser.add_argument("-v_freq", "--val_freq", type=int, default=10)
    parser.add_argument(
        "-i",
        "--input",
        default="/home/andres/Documents/workspace/disk-workspace/Datasets/radiology/brain/series_selection/train/",
    )
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-batch", "--train_batch_size", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    args = parser.parse_args()

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = glob.glob(os.path.join(args.input, "*/*"))

    # 4 classes classification system
    labels = []
    for l in images:
        mod = l.split("/")[-1]
        if mod == "FLAIR":
            labels.append(0)
        elif mod == "T1w":
            labels.append(1)
        elif mod == "T1wCE":
            labels.append(2)
        elif mod == "T2w":
            labels.append(3)

    labels = np.array(labels, dtype=np.int64)
    images = images[:50]
    labels = labels[:50]
    val_samples = int(len(labels) * args.val)
    train_files = [{"img": img, "label": label} for img, label in zip(images[:val_samples], labels[:val_samples])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-val_samples:], labels[-val_samples:])]

    num_labels = len(np.unique(labels))

    # define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys="img", reader="ITKReader", ensure_channel_first=True),
            NormalizeIntensityd(keys="img"),
            Resized(keys=["img"], spatial_size=(128, 128, 128)),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys="img", reader="ITKReader", ensure_channel_first=True),
            NormalizeIntensityd(keys="img"),
            Resized(keys="img", spatial_size=(128, 128, 128)),
        ]
    )

    # create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_labels).to(device)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), args.learning_rate)

    # Ignite trainer expects batch=(img, label) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["label"]), device, non_blocking)

    trainer = create_supervised_trainer(net, opt, loss, device, False, prepare_batch=prepare_batch)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = ModelCheckpoint("./runs_dict/", "net", n_saved=args.val_freq, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={"net": net, "opt": opt}
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    # add evaluation metric to the evaluator engine
    val_metrics = {"AUC": ROCAUC()}

    post_label = Compose([AsDiscrete(to_onehot=num_labels)])
    post_pred = Compose([Activations(softmax=True)])
    # Ignite evaluator expects batch=(img, label) and returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y, detach=False)],
        ),
    )

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_tensorboard_stats_handler.attach(evaluator)

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(patience=4, score_function=stopping_fn_from_metric("AUC"), trainer=trainer)
    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

    # create a validation data loader
    val_ds = monai.data.SmartCacheDataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    @trainer.on(Events.EPOCH_COMPLETED(every=args.val_freq))
    def run_validation(engine):
        evaluator.run(val_loader)

    # create a training data loader
    train_ds = monai.data.SmartCacheDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()
    )

    state = trainer.run(train_loader, args.epochs)
    print(state)


if __name__ == "__main__":
    main()
