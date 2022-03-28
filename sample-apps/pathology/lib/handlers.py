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
import math
import statistics
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch
from monai.config import IgniteInfo
from monai.metrics import compute_meandice
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
nib, _ = optional_import("nibabel")
torchvision, _ = optional_import("torchvision")
make_grid, _ = optional_import("torchvision.utils", name="make_grid")
Image, _ = optional_import("PIL.Image")
ImageDraw, _ = optional_import("PIL.ImageDraw")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")


class RegionDice:
    def __init__(self):
        self.data = []

    def reset(self):
        self.data = []

    def update(self, y_pred, y):
        y_pred = y_pred if torch.is_tensor(y_pred) else torch.from_numpy(y_pred)
        y = y if torch.is_tensor(y) else torch.from_numpy(y)

        score = compute_meandice(y_pred=y_pred, y=y, include_background=True).mean().item()
        if not math.isnan(score):
            self.data.append(score)

    def mean(self):
        return statistics.mean(self.data) if len(self.data) else 0

    def stdev(self):
        return statistics.stdev(self.data) if len(self.data) > 1 else 0


class TensorBoardImageHandler:
    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        tag_name="val_acc",
        interval: int = 1,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        batch_limit=1,
        device=None,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.tag_name = tag_name
        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.batch_limit = batch_limit
        self.device = device

        self.logger = logging.getLogger(__name__)

        if torch.distributed.is_initialized():
            self.tag_name = "{}-r{}".format(self.tag_name, torch.distributed.get_rank())
        self.metric_data: Dict[Any, Any] = dict()

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self, "iteration")
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self, "epoch")

    def __call__(self, engine: Engine, action) -> None:
        epoch = engine.state.epoch
        batch_data = self.batch_transform(engine.state.batch)
        output_data = self.output_transform(engine.state.output)

        if action == "iteration":
            for bidx in range(len(batch_data)):
                y = batch_data[bidx]["label"].detach().cpu().numpy()
                y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

                for region in range(y_pred.shape[0]):
                    if region == 0 and y_pred.shape[0] != y.shape[0] and y_pred.shape[0] > 1:  # one-hot; background
                        continue

                    if self.metric_data.get(region) is None:
                        self.metric_data[region] = RegionDice()

                    if y_pred.shape[0] == y.shape[0]:
                        label = y[region][np.newaxis]
                    else:
                        label = np.zeros(y.shape)
                        label[y == region] = region

                    self.metric_data[region].update(
                        y_pred=y_pred[region][np.newaxis],
                        y=label,
                    )
            return

        self.write_region_metrics(epoch)
        self.write_images(batch_data, output_data, epoch)

    def write_images(self, batch_data, output_data, epoch):
        for bidx in range(len(batch_data)):
            image = batch_data[bidx]["image"].detach().cpu().numpy()
            y = batch_data[bidx]["label"].detach().cpu().numpy()
            y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

            # Only consider non-empty label in case single write
            if self.batch_limit == 1 and bidx < (len(batch_data) - 1) and np.sum(y) == 0:
                continue

            tag_prefix = f"b{bidx} - " if self.batch_limit != 1 else ""
            img_tensor = make_grid(torch.from_numpy(image[:3] * 128 + 128), normalize=True)
            self.writer.add_image(tag=f"{tag_prefix}Image", img_tensor=img_tensor, global_step=epoch)

            for region in range(y_pred.shape[0]):
                if region == 0 and y_pred.shape[0] != y.shape[0] and y_pred.shape[0] > 1:  # one-hot; background
                    continue

                if y_pred.shape[0] == y.shape[0]:
                    label = y[region][np.newaxis]
                else:
                    label = np.zeros(y.shape)
                    label[y == region] = region

                self.logger.info(
                    "{} - {} - Image: {}; Label: {} (nz: {}); Pred: {} (nz: {})".format(
                        bidx,
                        region,
                        image.shape,
                        label.shape,
                        np.count_nonzero(label),
                        y_pred.shape,
                        np.count_nonzero(y_pred[region]),
                    )
                )

                tag_prefix = f"b{bidx}:l{region} - " if self.batch_limit != 1 else f"l{region} - "

                label_pred = [label, y_pred[region][None]]
                label_pred_tag = f"{tag_prefix}Label vs Pred:"
                if image.shape[0] == 5:
                    label_pred = [label, y_pred[region][None], image[3][None], image[4][None]]
                    label_pred_tag = f"{tag_prefix}Label vs Pred vs Pos vs Neg"

                img_tensor = make_grid(
                    tensor=torch.from_numpy(np.array(label_pred)),
                    nrow=4,
                    normalize=True,
                    pad_value=10,
                )
                self.writer.add_image(tag=label_pred_tag, img_tensor=img_tensor, global_step=epoch)

            if self.batch_limit == 1 or bidx == (self.batch_limit - 1):
                break

    def write_region_metrics(self, epoch):
        metric_sum = 0
        for region in self.metric_data:
            metric = self.metric_data[region].mean()
            self.logger.info(
                "Epoch[{}] Metrics -- Region: {:0>2d}, {}: {:.4f}".format(epoch, region, self.tag_name, metric)
            )

            self.writer.add_scalar("dice_{:0>2d}".format(region), metric, epoch)
            metric_sum += metric

        if len(self.metric_data) > 1:
            metric_avg = metric_sum / len(self.metric_data)
            self.writer.add_scalar("dice_regions_avg", metric_avg, epoch)

        self.writer.flush()
        self.metric_data = {}
