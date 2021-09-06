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

import numpy as np
import torch
from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys


class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for DeepEdit Training/Evaluation.

    Args:
        deepgrow_probability: probability of simulating clicks in an iteration
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        max_interactions: maximum number of click interactions per iteration if deepgrow training invoked for iteration
        train: True for training mode or False for evaluation mode
        click_probability_key: key to click/interaction probability
    """

    def __init__(
        self,
        deepgrow_probability: float,
        transforms: Union[Sequence[Callable], Callable],
        max_interactions: int,
        train: bool,
        click_probability_key: str = "probability",
    ) -> None:

        if not isinstance(transforms, Compose):
            transforms = Compose(transforms)

        self.deepgrow_probability = deepgrow_probability
        self.transforms = transforms
        self.max_interactions = max_interactions
        self.train = train
        self.click_probability_key = click_probability_key

    def __call__(self, engine: Union[SupervisedTrainer, SupervisedEvaluator], batchdata: Dict[str, torch.Tensor]):

        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        pos_click_sum = 0
        neg_click_sum = 0
        if np.random.choice([True, False], p=[self.deepgrow_probability, 1 - self.deepgrow_probability]):
            pos_click_sum += 1  # increase pos_click_sum by 1-click for AddInitialSeedPointd pre_transform
            for j in range(self.max_interactions):

                # print("Inner iteration (click simulations running): ", str(j))

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

                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)

                for i in range(len(batchdata_list)):
                    batchdata_list[i][self.click_probability_key] = (
                        (1.0 - ((1.0 / self.max_interactions) * j)) if self.train else 1.0
                    )
                    batchdata_list[i] = self.transforms(batchdata_list[i])

                batchdata = list_data_collate(batchdata_list)

                # first item in batch only
                pos_click_sum += (batchdata_list[0]["is_pos"]) * 1
                neg_click_sum += (batchdata_list[0]["is_neg"]) * 1

                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        else:
            # zero out input guidance channels
            batchdata_list = decollate_batch(batchdata, detach=True)
            for i in range(len(batchdata_list)):
                batchdata_list[i][CommonKeys.IMAGE][-1] *= 0
                batchdata_list[i][CommonKeys.IMAGE][-2] *= 0
            batchdata = list_data_collate(batchdata_list)

        # first item in batch only
        engine.state.batch = batchdata
        engine.state.batch.update({"pos_click_sum": torch.tensor(pos_click_sum)})
        engine.state.batch.update({"neg_click_sum": torch.tensor(neg_click_sum)})

        return engine._iteration(engine, batchdata)
