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
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from monai.config import IgniteInfo
from monai.transforms import rescale_array
from monai.utils import min_version, optional_import
from monai.visualize import plot_2d_or_3d_image

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


class TensorBoardImageHandler:
    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        tag_name="val_acc",
        interval: int = 1,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.tag_name = tag_name
        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = logging.getLogger(__name__)

        if torch.distributed.is_initialized():
            self.tag_name = "{}-r{}".format(self.tag_name, torch.distributed.get_rank())

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        epoch = engine.state.epoch
        image_grid = rescale_array(self.batch_transform(engine.state.batch)[0]["image"].detach().cpu().numpy(), 0, 1)
        label_grid = rescale_array(self.batch_transform(engine.state.batch)[0]["label"].detach().cpu().numpy(), 0, 1)
        pred_grid = rescale_array(self.output_transform(engine.state.output)[0]["pred"].detach().cpu().numpy(), 0, 1)
        self.logger.info(f"ImageGrid: {image_grid.shape}; LabelGrid: {label_grid.shape}; PredGrid: {pred_grid.shape}")

        patch_size = image_grid.shape[-1]
        patch_per_side = int(math.sqrt(image_grid.shape[0]))

        image = np.zeros((3, patch_size * patch_per_side, patch_size * patch_per_side), dtype=image_grid.dtype)
        label = np.zeros((patch_size * patch_per_side, patch_size * patch_per_side), dtype=image_grid.dtype)
        pred = np.zeros((patch_size * patch_per_side, patch_size * patch_per_side), dtype=image_grid.dtype)

        count = 0
        for x_idx in range(patch_per_side):
            for y_idx in range(patch_per_side):
                x_start = x_idx * patch_size
                x_end = x_start + patch_size
                y_start = y_idx * patch_size
                y_end = y_start + patch_size

                image[:, x_start:x_end, y_start:y_end] = image_grid[count]
                label[x_start:x_end, y_start:y_end] = label_grid[count]
                pred[x_start:x_end, y_start:y_end] = pred_grid[count]
                count += 1

        label = label[np.newaxis]
        pred = pred[np.newaxis]
        self.logger.info(f"Image: {image.shape}; Label: {label.shape}; Pred: {pred.shape}")

        plot_2d_or_3d_image(data=image[None], step=epoch, max_channels=3, writer=self.writer, tag=f"Image")
        plot_2d_or_3d_image(data=label[None], step=epoch, max_channels=3, writer=self.writer, tag=f"Label")
        plot_2d_or_3d_image(data=pred[None], step=epoch, max_channels=3, writer=self.writer, tag=f"Pred")
