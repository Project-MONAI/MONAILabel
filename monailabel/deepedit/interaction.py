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

from typing import Callable, Dict, Sequence, Union

import torch
from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys


class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for Deepgrow Training/Evaluation.
    This implementation is based on:

        Sakinis et al., Interactive segmentation of medical images through
        fully convolutional neural networks. (2019) https://arxiv.org/abs/1903.08205

    Args:
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        max_interactions: maximum number of interactions per iteration
        train: training or evaluation
        key_probability: field name to fill probability for every interaction
    """

    def __init__(
        self,
        transforms: Union[Sequence[Callable], Callable],
        max_interactions: int,
        train: bool,
        key_probability: str = "probability",
    ) -> None:

        if not isinstance(transforms, Compose):
            transforms = Compose(transforms)

        self.transforms = transforms
        self.max_interactions = max_interactions
        self.train = train
        self.key_probability = key_probability

    def __call__(self, engine: Union[SupervisedTrainer, SupervisedEvaluator], batchdata: Dict[str, torch.Tensor]):

        # batchdata.update({'inner_iter': torch.tensor([0])})

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        for j in range(self.max_interactions):
            inputs, _ = engine.prepare_batch(batchdata)
            inputs = inputs.to(engine.state.device)

            engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)

            engine.network.eval()
            with torch.no_grad():
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions = engine.inferer(inputs, engine.network)
                else:
                    predictions = engine.inferer(inputs, engine.network)

            batchdata.update({CommonKeys.PRED: predictions})

            # decollate batch data to execute click transforms
            batchdata_list = decollate_batch(batchdata, detach=True)

            for i in range(len(batchdata_list)):  # Only 1 iteration
                batchdata_list[i][self.key_probability] = (
                    (1.0 - ((1.0 / self.max_interactions) * j)) if self.train else 1.0
                )
                batchdata_list[i] = self.transforms(batchdata_list[i])

            # collate list into a batch for next round interaction
            batchdata = list_data_collate(batchdata_list)

            # Why 'inner_iter' isn't visible or always ZERO in the Tensorboard handler??
            # I was modifying batchdata without modifying the state!!!

            engine.state.batch.update({"inner_iter": j})
            engine.state.batch.update({"img_inner_iter": batchdata_list})
            engine.state.batch.update({"is_pos": batchdata_list[0]["is_pos"]})
            engine.state.batch.update({"is_neg": batchdata_list[0]["is_neg"]})
            engine.state.batch.update({"max_iter": self.max_interactions})

            engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

            # Need to remove these from dictionary as collating shows errors
            engine.state.batch.pop("img_inner_iter")

        return engine._iteration(engine, batchdata)
