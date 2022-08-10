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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch
from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

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


class TensorBoardImageHandler:
    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        tag_prefix="",
        interval: int = 1,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        batch_limit=1,
        device=None,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.tag_prefix = tag_prefix
        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.batch_limit = batch_limit
        self.device = device

        self.logger = logging.getLogger(__name__)

        if torch.distributed.is_initialized():
            self.tag_prefix = f"{self.tag_prefix}r{torch.distributed.get_rank()}-"
        self.metric_data: Dict[Any, Any] = dict()

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self, "epoch")

    def __call__(self, engine: Engine, action) -> None:
        epoch = engine.state.epoch
        batch_data = self.batch_transform(engine.state.batch)
        output_data = self.output_transform(engine.state.output)

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
            tag_prefix = f"{self.tag_prefix}{tag_prefix}"
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
                    "{} - {} - Image: {}; Label: {} (nz: {}); Pred: {} (nz: {}); Sig: (pos-nz: {}, neg-nz: {})".format(
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
                tag_prefix = f"{self.tag_prefix}{tag_prefix}"

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
