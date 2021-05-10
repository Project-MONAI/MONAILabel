from typing import Dict, Union

import torch
from monai.apps.deepgrow.interaction import Interaction
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.utils.enums import CommonKeys

from .events import DeepEditEvents


class DeepEditInteraction(Interaction):
    def __call__(self, engine: Union[SupervisedTrainer, SupervisedEvaluator], batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        for j in range(self.max_interactions):
            inputs, _ = engine.prepare_batch(batchdata)
            inputs = inputs.to(engine.state.device)

            engine.fire_event(DeepEditEvents.INNER_ITERATION_STARTED)

            engine.network.eval()
            with torch.no_grad():
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions = engine.inferer(inputs, engine.network)
                else:
                    predictions = engine.inferer(inputs, engine.network)

            engine.state.batch_data_inner = batchdata
            engine.state.preds_inner = predictions
            engine.state.step_inner = j

            engine.fire_event(DeepEditEvents.INNER_ITERATION_COMPLETED)

            batchdata.update({CommonKeys.PRED: predictions})
            batchdata[self.key_probability] = torch.as_tensor(
                ([1.0 - ((1.0 / self.max_interactions) * j)] if self.train else [1.0]) * len(inputs)
            )
            batchdata = self.transforms(batchdata)

        return engine._iteration(engine, batchdata)
