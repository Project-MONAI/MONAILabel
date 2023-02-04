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

import glob
import tempfile
import unittest

import numpy as np
import torch
from ignite.engine import Engine, Events
from monai.utils import optional_import
from parameterized import parameterized

from monailabel.deepedit.handlers import TensorBoard2DImageHandler, TensorBoardImageHandler

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")

TEST_CASES = [[[20, 20]]]


@unittest.skipUnless(has_tb, "Requires SummaryWriter installation")
class TestHandlerTBImage(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_tb_image_shape(self, shape):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                engine.state.batch = batch
                return [{"pred": np.random.randint(0, 2, size=(10, 4, *shape))}]

            engine = Engine(_train_func)
            # set up testing handler
            stats_handler = TensorBoardImageHandler(
                log_dir=tempdir,
            )
            engine.add_event_handler(Events.ITERATION_COMPLETED, stats_handler)

            data = zip(
                [
                    {
                        "image": torch.as_tensor(np.random.normal(size=(10, 4, *shape))),
                        "image_meta_dict": {"filename_or_obj": "test_image1.nii.gz"},
                        "label": np.random.randint(0, 2, size=(10, 4, *shape)),
                    },
                    {
                        "image": torch.as_tensor(np.random.normal(size=(10, 4, *shape))),
                        "image_meta_dict": {"filename_or_obj": "test_image2.nii.gz"},
                        "label": np.random.randint(0, 2, size=(10, 4, *shape)),
                    },
                ]
            )
            engine.run(data, epoch_length=10, max_epochs=1)
            stats_handler.close()

            self.assertTrue(len(glob.glob(tempdir)) > 0)


@unittest.skipUnless(has_tb, "Requires SummaryWriter installation")
class TestHandlerTB2DImage(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_tb_2dimage_shape(self, shape):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                engine.state.batch = batch

                return [{"pred": torch.tensor(np.zeros((1, *shape)))}]

            engine = Engine(_train_func)
            # set up testing handler
            stats_handler = TensorBoard2DImageHandler(log_dir=tempdir)

            engine.add_event_handler(Events.ITERATION_COMPLETED, stats_handler, "iteration")

            data = zip(
                [
                    {
                        "image": torch.as_tensor(np.zeros((1, 3, *shape))),
                        "image_meta_dict": {"filename_or_obj": "test_image1.nii.gz"},
                        "label": torch.as_tensor(np.zeros((1, *shape))),
                    },
                    {
                        "image": torch.as_tensor(np.zeros((1, 3, *shape))),
                        "image_meta_dict": {"filename_or_obj": "test_image2.nii.gz"},
                        "label": torch.as_tensor(np.zeros((1, *shape))),
                    },
                ]
            )
            engine.run(data, epoch_length=10, max_epochs=1)
            self.assertTrue(len(glob.glob(tempdir)) > 0)


if __name__ == "__main__":
    unittest.main()
