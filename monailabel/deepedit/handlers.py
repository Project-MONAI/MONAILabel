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

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from monai.config import IgniteInfo
from monai.utils import min_version, optional_import
from monai.visualize import plot_2d_or_3d_image

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
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.num_total_clicks = 0
        self.cumulative_pos_clicks = 0
        self.cumulative_neg_clicks = 0

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
                
        """
        Adding simulated clicks stats
        """
        self.num_pos_clicks = self.batch_transform(engine.state.batch)[0]["pos_click_sum"]
        self.cumulative_pos_clicks += self.batch_transform(engine.state.batch)[0]["pos_click_sum"]

        self.num_neg_clicks = self.batch_transform(engine.state.batch)[0]["neg_click_sum"]
        self.cumulative_neg_clicks += self.batch_transform(engine.state.batch)[0]["neg_click_sum"]
            
        self.num_total_clicks = self.num_pos_clicks + self.num_neg_clicks
            
        self._writer.add_scalar("Positive clicks", self.num_pos_clicks, step)
        self._writer.add_scalar("Negative clicks", self.num_neg_clicks, step)
        self._writer.add_scalar("Total clicks", self.num_total_clicks, step)
        self._writer.add_scalar("Positive clicks (cumulative)", self.cumulative_pos_clicks, step)
        self._writer.add_scalar("Negative clicks (cumulative)", self.cumulative_neg_clicks, step)
        
        print("handler_pos_clicks", self.num_pos_clicks)
        print("handler_neg_clicks", self.num_neg_clicks)
        
        """
        IMAGE
        """
        print("Handler in operation")
        show_image = self.batch_transform(engine.state.batch)[0]["image"][0, ...][None]
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
                show_image[None],
                step,
                self._writer,
                0,
                self.max_channels,
                self.max_frames,
                "step_" + str(step) + "_image_" + filename,
            )

        """
        LABEL
        """
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
                show_label[None],
                step,
                self._writer,
                0,
                self.max_channels,
                self.max_frames,
                "step_" + str(step) + "_label_" + filename,
            )
            
        """
        PREDICTION
        """
        show_prediction = self.output_transform(engine.state.output[0])["pred"][0, ...][None]
        if isinstance(show_prediction, torch.Tensor):
            show_prediction = show_prediction.detach().cpu().numpy()
        if show_prediction is not None:
            if not isinstance(show_prediction, np.ndarray):
                raise TypeError(
                    "show_pred must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_label).__name__}."
                )
            show_prediction = (show_prediction > 0.5).astype(np.float32) # binarize
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                show_prediction[None],
                step,
                self._writer,
                0,
                self.max_channels,
                self.max_frames,
                "step_" + str(step) + "_prediction_" + filename,
            )
        
        """
        POSITIVE CLICKS
        """
        show_pos_clicks = self.batch_transform(engine.state.batch)[0]["image"][1, ...][None]
        if isinstance(show_pos_clicks, torch.Tensor):
            show_pos_clicks = show_pos_clicks.detach().cpu().numpy()
        print("handler_pos_click_image_sum", np.sum(show_pos_clicks))
        if show_pos_clicks is not None:
            if not isinstance(show_pos_clicks, np.ndarray):
                raise TypeError(
                    "show_pos_clicks must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_pos_clicks).__name__}."
                )
            #show_pos_clicks = show_label + show_pos_clicks
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                show_pos_clicks[None],
                step,
                self._writer,
                0,
                self.max_channels,
                self.max_frames,
                "step_" + str(step) + "_pos_clicks_" + filename,
            )

        """
        NEGATIVE CLICKS
        """
        #show_neg_clicks = self.batch_transform(engine.state.batch)["img_inner_iter"][0]["image"][2, ...][None]
        show_neg_clicks = self.batch_transform(engine.state.batch)[0]["image"][2, ...][None]
        if isinstance(show_neg_clicks, torch.Tensor):
            show_neg_clicks = show_neg_clicks.detach().cpu().numpy()
        print("handler_neg_click_image_sum", np.sum(show_neg_clicks))
        if show_neg_clicks is not None:
            if not isinstance(show_neg_clicks, np.ndarray):
                raise TypeError(
                    "show_neg_clicks must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_neg_clicks).__name__}."
                )
            #show_neg_clicks = show_label + show_neg_clicks
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                show_neg_clicks[None],
                step,
                self._writer,
                0,
                self.max_channels,
                self.max_frames,
                "step_" + str(step) + "_neg_clicks_" + filename,
            )

        self._writer.flush()

    def close(self):
        """
        Close the summary writer if created in this TensorBoard handler.

        """
        if self.internal_writer:
            self._writer.close()
