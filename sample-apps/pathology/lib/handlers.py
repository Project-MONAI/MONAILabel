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
import math
import statistics
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
from monai.config import IgniteInfo
from monai.metrics import compute_meandice
from monai.utils import min_version, optional_import
from sklearn.metrics import classification_report

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
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
        class_names=None,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.tag_name = tag_name
        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.batch_limit = batch_limit
        self.device = device
        self.class_names = class_names

        self.logger = logging.getLogger(__name__)

        if torch.distributed.is_initialized():
            self.tag_name = f"{self.tag_name}-r{torch.distributed.get_rank()}"
        self.metric_data: Dict[Any, Any] = dict()
        self.class_y: List[Any] = []
        self.class_y_pred: List[Any] = []

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self, "iteration")
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self, "epoch")

    def __call__(self, engine: Engine, action) -> None:
        epoch = engine.state.epoch
        batch_data = self.batch_transform(engine.state.batch)
        output_data = self.output_transform(engine.state.output)

        if action == "iteration":
            for bidx in range(len(batch_data)):
                if self.class_names:
                    y = output_data[bidx]["label"].detach().cpu().numpy()
                    y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

                    self.class_y.append(np.argmax(y))
                    self.class_y_pred.append(np.argmax(y_pred))
                    continue

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
            y = output_data[bidx]["label"].detach().cpu().numpy()

            tag_prefix = f"b{bidx} - " if self.batch_limit != 1 else ""
            img_tensor = make_grid(torch.from_numpy(image[:3] * 128 + 128), normalize=True)
            self.writer.add_image(tag=f"{tag_prefix}Image", img_tensor=img_tensor, global_step=epoch)

            if self.class_names:
                sig_np = image[:3] * 128 + 128
                sig_np[0, :, :] = np.where(image[3] > 0, 1, sig_np[0, :, :])
                sig_tensor = make_grid(torch.from_numpy(sig_np), normalize=True)
                self.writer.add_image(tag=f"{tag_prefix}Signal", img_tensor=sig_tensor, global_step=epoch)
                if np.count_nonzero(image[3]) == 0:
                    self.logger.info("+++++++++ BUG (Signal is ZERO)")

                y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

                y_c = np.argmax(y)
                y_pred_c = np.argmax(y_pred)

                tag_prefix = f"b{bidx} - " if self.batch_limit != 1 else ""
                label_pred_tag = f"{tag_prefix}Label vs Pred:"

                y_img = Image.new("RGB", (200, 100))
                draw = ImageDraw.Draw(y_img)
                draw.text((10, 50), self.class_names.get(f"{y_c}", f"{y_c}"))

                y_pred_img = Image.new("RGB", (200, 100), "green" if y_c == y_pred_c else "red")
                draw = ImageDraw.Draw(y_pred_img)
                draw.text((10, 50), self.class_names.get(f"{y_pred_c}", f"{y_pred_c}"))

                label_pred = [np.moveaxis(np.array(y_img), -1, 0), np.moveaxis(np.array(y_pred_img), -1, 0)]
                img_tensor = make_grid(
                    tensor=torch.from_numpy(np.array(label_pred)),
                    nrow=3,
                    normalize=False,
                    pad_value=10,
                )
                self.writer.add_image(tag=label_pred_tag, img_tensor=img_tensor, global_step=epoch)
            else:
                # Only consider non-empty label in case single write
                if self.batch_limit == 1 and bidx < (len(batch_data) - 1) and np.sum(y) == 0:
                    continue

                y = batch_data[bidx]["label"].detach().cpu().numpy()
                y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

                for region in range(y_pred.shape[0]):
                    if region == 0 and y_pred.shape[0] != y.shape[0] and y_pred.shape[0] > 1:  # one-hot; background
                        continue

                    if y_pred.shape[0] == y.shape[0]:
                        label = y[region][np.newaxis]
                    else:
                        label = np.zeros(y.shape)
                        label[y == region] = region

                    self.logger.info(
                        "{} - {} - Image: {};"
                        " Label: {} (nz: {});"
                        " Pred: {} (nz: {});"
                        " Sig: (pos-nz: {}, neg-nz: {})".format(
                            bidx,
                            region,
                            image.shape,
                            label.shape,
                            np.count_nonzero(label),
                            y_pred.shape,
                            np.count_nonzero(y_pred[region]),
                            np.count_nonzero(image[3]) if image.shape[0] == 5 else 0,
                            np.count_nonzero(image[4]) if image.shape[0] == 5 else 0,
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
        if self.class_names and len(self.class_y):
            cr = classification_report(self.class_y, self.class_y_pred, output_dict=True, zero_division=0)
            for k, v in cr.items():
                if isinstance(v, dict):
                    ltext = []
                    for n, m in v.items():
                        ltext.append(f"{n} => {m:.4f}")
                        cname = self.class_names.get(k, k)
                        self.writer.add_scalar(f"cr_{k}_{n}", m, epoch)
                        self.logger.info(f"Epoch[{epoch}] Metrics -- Class: {cname}; {'; '.join(ltext)}")
                else:
                    self.logger.info(f"Epoch[{epoch}] Metrics -- {k} => {v:.4f}")
                    self.writer.add_scalar(f"cr_{k}", v, epoch)

            self.class_y = []
            self.class_y_pred = []
            return

        if len(self.metric_data) > 1:
            metric_sum = 0
            for region in self.metric_data:
                metric = self.metric_data[region].mean()
                self.logger.info(f"Epoch[{epoch}] Metrics -- Region: {region:0>2d}, {self.tag_name}: {metric:.4f}")

                self.writer.add_scalar(f"dice_{region:0>2d}", metric, epoch)
                metric_sum += metric

            metric_avg = metric_sum / len(self.metric_data)
            self.writer.add_scalar("dice_regions_avg", metric_avg, epoch)

        self.writer.flush()
        self.metric_data = {}
