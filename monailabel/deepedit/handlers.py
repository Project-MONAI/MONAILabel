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
from monai.visualize import plot_2d_or_3d_image
from torch import distributed as dist
from torchvision.utils import make_grid

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")


class TensorBoardImageHandler:
    """
    TensorBoardImageHandler is an Ignite Event handler that can visualize images, labels and outputs as 2D/3D images.
    2D output (shape in Batch, channel, H, W) will be shown as simple image using the first element in the batch,
    for 3D to ND output (shape in Batch, channel, H, W, D) input, each of ``self.max_channels`` number of images'
    last three dimensions will be shown as animated GIF along the last axis (typically Depth).

    It can be used for any Ignite Engine (trainer, validator and evaluator).
    User can easily add it to engine for any expected Event, for example: ``EPOCH_COMPLETED``,
    ``ITERATION_COMPLETED``. The expected data source is ignite's ``engine.state.batch`` and ``engine.state.output``.

    Default behavior:
        - Show y_pred as images (GIF for 3D) on TensorBoard when Event triggered,
        - Need to use ``batch_transform`` and ``output_transform`` to specify
          how many images to show and show which channel.
        - Expects ``batch_transform(engine.state.batch)`` to return data
          format: (image[N, channel, ...], label[N, channel, ...]).
        - Expects ``output_transform(engine.state.output)`` to return a torch
          tensor in format (y_pred[N, channel, ...], loss).

    """

    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        interval: int = 1,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        max_frames: int = 64,
    ) -> None:
        """
        Args:
            summary_writer: user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            log_dir: if using default SummaryWriter, write logs to this directory, default is `./runs`.
            interval: plot content from engine.state every N epochs or every N iterations, default is 1.
            batch_transform: a callable that is used to extract `image` and `label` from `ignite.engine.state.batch`,
                then construct `(image, label)` pair. for example: if `ignite.engine.state.batch` is `{"image": xxx,
                "label": xxx, "other": xxx}`, `batch_transform` can be `lambda x: (x["image"], x["label"])`.
                will use the result to plot image from `result[0][index]` and plot label from `result[1][index]`.
            output_transform: a callable that is used to extract the `predictions` data from
                `ignite.engine.state.output`, will use the result to plot output from `result[index]`.
            global_iter_transform: a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
            index: plot which element in a data batch, default is the first element.
            max_channels: number of channels to plot.
            max_frames: number of frames for 2D-t plot.
        """

        if summary_writer is None:
            self._writer = SummaryWriter(log_dir=log_dir)
            self.internal_writer = True
        else:
            self._writer = summary_writer
            self.internal_writer = False

        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_frames = max_frames
        self.max_channels = max_channels

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        Raises:
            TypeError: When ``output_transform(engine.state.output)[0]`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
            TypeError: When ``batch_transform(engine.state.batch)[1]`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
            TypeError: When ``output_transform(engine.state.output)`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.

        """
        step = self.global_iter_transform(engine.state.iteration)
        filename = (
            self.batch_transform(engine.state.batch)[0]["image_meta_dict"]["filename_or_obj"]
            .split("/")[-1]
            .split(".")[0]
        )

        input_tensor = self.batch_transform(engine.state.batch)[0]["image"]

        # IMAGE
        show_image = input_tensor[0, ...][None]
        if isinstance(show_image, torch.Tensor):
            show_image = show_image.detach().cpu().numpy()
        if show_image is not None:
            if not isinstance(show_image, np.ndarray):
                raise TypeError(
                    "show_image must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_image).__name__}."
                )
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_image[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                max_frames=self.max_frames,
                tag="step_" + str(step) + "_image_" + filename,
            )

        # LABEL
        show_label = self.batch_transform(engine.state.batch)[0]["label"][0, ...][None]
        if isinstance(show_label, torch.Tensor):
            show_label = show_label.detach().cpu().numpy()
        if show_label is not None:
            if not isinstance(show_label, np.ndarray):
                raise TypeError(
                    "show_label must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
                )
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_label[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                max_frames=self.max_frames,
                tag="step_" + str(step) + "_label_" + filename,
            )

        # PREDICTION
        all_preds = self.output_transform(engine.state.output)[0]["pred"]
        for idx in range(all_preds.shape[0]):
            show_prediction = all_preds[idx, ...][None]
            if isinstance(show_prediction, torch.Tensor):
                show_prediction = show_prediction.detach().cpu().numpy()
            if show_prediction is not None:
                if not isinstance(show_prediction, np.ndarray):
                    raise TypeError(
                        "show_pred must be None or one of "
                        f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
                    )
                plot_2d_or_3d_image(
                    # add batch dim and plot the first item
                    data=show_prediction[None],
                    step=step,
                    writer=self._writer,
                    index=0,
                    max_channels=self.max_channels,
                    max_frames=self.max_frames,
                    tag="step_" + str(step) + f"_prediction_for_label_{str(idx)}_" + filename,
                )

        # ALL CLICKS
        show_pos_clicks = input_tensor[1:, ...][None]
        if isinstance(show_pos_clicks, torch.Tensor):
            show_pos_clicks = show_pos_clicks.detach().cpu().numpy()
            # Adding all labels in a single channel tensor
            if show_pos_clicks.shape[1] > 1:
                show_pos_clicks = np.sum(show_pos_clicks, axis=1)
        if show_pos_clicks is not None:
            if not isinstance(show_pos_clicks, np.ndarray):
                raise TypeError(
                    "show_pos_clicks must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_pos_clicks).__name__}."
                )
            show_pos_clicks = show_label * (1 - show_pos_clicks)
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_pos_clicks[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                max_frames=self.max_frames,
                tag="step_" + str(step) + "_all_clicks_" + filename,
            )

        self._writer.flush()

    def close(self):
        """
        Close the summary writer if created in this TensorBoard handler.

        """
        if self.internal_writer:
            self._writer.close()


class TensorBoard2DImageHandler:
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

        if dist.is_initialized():
            self.tag_prefix = f"{self.tag_prefix}r{dist.get_rank()}-"
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
