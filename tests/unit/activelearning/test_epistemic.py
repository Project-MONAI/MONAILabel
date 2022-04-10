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

import unittest
from functools import partial

import numpy as np
import torch
import tqdm
from monai.data import CacheDataset, DataLoader, create_test_image_3d
from monai.data.utils import pad_list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel,
    AddChanneld,
    Compose,
    CropForeground,
    CropForegroundd,
    DivisiblePad,
    DivisiblePadd,
)
from monai.utils import set_determinism

from monailabel.tasks.scoring.epistemic import EpistemicScoring

trange = partial(tqdm.trange, desc="training")


class TestEpistemicScoring(unittest.TestCase):
    @staticmethod
    def get_data(num_examples, input_size, data_type=np.asarray, include_label=True):
        custom_create_test_image_3d = partial(
            create_test_image_3d, *input_size, rad_max=7, num_seg_classes=1, num_objs=1
        )
        data = []
        for _ in range(num_examples):
            im, label = custom_create_test_image_3d()
            d = {}
            d["image"] = data_type(im)
            d["image_meta_dict"] = {"affine": np.eye(4)}
            if include_label:
                d["label"] = data_type(label)
                d["label_meta_dict"] = {"affine": np.eye(4)}
            d["label_transforms"] = []
            data.append(d)
        return data[0] if num_examples == 1 else data

    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_epistemic_scoring(self):
        input_size = (20, 20, 20)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        keys = ["image", "label"]
        num_training_ims = 10
        train_data = self.get_data(num_training_ims, input_size)
        test_data = self.get_data(1, input_size)

        transforms = Compose(
            [
                AddChanneld(keys),
                CropForegroundd(keys, source_key="image"),
                DivisiblePadd(keys, 4),
            ]
        )

        infer_transforms = Compose(
            [
                AddChannel(),
                CropForeground(),
                DivisiblePad(4),
            ]
        )

        train_ds = CacheDataset(train_data, transforms)
        # output might be different size, so pad so that they match
        train_loader = DataLoader(train_ds, batch_size=2, collate_fn=pad_list_data_collate)

        model = UNet(3, 1, 1, channels=(6, 6), strides=(2, 2)).to(device)
        loss_function = DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        num_epochs = 10
        for _ in trange(num_epochs):
            epoch_loss = 0

            for batch_data in train_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

        entropy_score = EpistemicScoring(
            model=model, transforms=infer_transforms, roi_size=[20, 20, 20], num_samples=10
        )
        # Call Individual Infer from Epistemic Scoring
        ip_stack = [test_data["image"], test_data["image"], test_data["image"]]
        ip_stack = np.array(ip_stack)
        score_3d = entropy_score.entropy_3d_volume(ip_stack)
        score_3d_sum = np.sum(score_3d)
        # Call Entropy Metric from Epistemic Scoring
        self.assertEqual(score_3d.shape, input_size)
        self.assertIsInstance(score_3d_sum, np.float32)
        self.assertGreater(score_3d_sum, 3.0)


if __name__ == "__main__":
    unittest.main()
