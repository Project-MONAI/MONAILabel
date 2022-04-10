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

import datetime
import filecmp
import json
import logging
import os
import shutil
import time
from typing import Any, Dict

import torch
from monai.engines.workflow import Engine, Events

logger = logging.getLogger(__name__)


def prepare_stats(start_ts, trainer, evaluator):
    def tensor_to_list(d):
        r = dict()
        for dk, dv in d.items():
            r[dk] = dv.tolist() if torch.is_tensor(dv) else dv
        return r

    stats: Dict[str, Any] = dict()
    stats.update(trainer.get_train_stats())
    stats["epoch"] = trainer.state.epoch
    stats["start_ts"] = int(start_ts)

    if trainer.state.epoch == trainer.state.max_epochs:
        stats["total_time"] = str(datetime.timedelta(seconds=int(time.time() - start_ts)))
    else:
        stats["current_time"] = str(datetime.timedelta(seconds=int(time.time() - start_ts)))

    for k, v in {"train": trainer, "eval": evaluator}.items():
        if not v:
            continue

        stats["best_metric"] = v.state.best_metric
        stats[k] = {
            "metrics": tensor_to_list(v.state.metrics),
            "key_metric_name": v.state.key_metric_name,
            "best_metric": v.state.best_metric,
            "best_metric_epoch": v.state.best_metric_epoch,
        }
    return stats


class PublishStatsAndModel:
    def __init__(self, stats_path, publish_path, key_metric_filename, start_ts, run_id, output_dir, trainer, evaluator):
        self._stats_path = stats_path
        self._publish_path = publish_path
        self._key_metric_filename = key_metric_filename
        self.start_ts = start_ts
        self.run_id = run_id
        self.output_dir = output_dir
        self.trainer = trainer
        self.evaluator = evaluator

    def iteration_completed(self):
        filename = datetime.datetime.now().strftime(f"stats_{self.run_id}.json")
        filename = os.path.join(self.output_dir, filename)

        stats = prepare_stats(self.start_ts, self.trainer, self.evaluator)
        with open(filename, "w") as f:
            json.dump(stats, f, indent=2)

        if self._stats_path:
            shutil.copy(filename, self._stats_path)

        publish_path = self._publish_path
        if publish_path:
            final_model = os.path.join(self.output_dir, self._key_metric_filename)
            if os.path.exists(final_model):
                if not os.path.exists(publish_path) or not filecmp.cmp(publish_path, final_model):
                    shutil.copy(final_model, publish_path)
                    logger.info(f"New Model published: {final_model} => {publish_path}")
        return stats

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self.iteration_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.iteration_completed)

    def __call__(self, engine: Engine) -> None:
        self.iteration_completed()
