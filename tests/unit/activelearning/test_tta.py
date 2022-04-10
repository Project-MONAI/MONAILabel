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
from monai.data import CacheDataset, DataLoader, create_test_image_2d
from monai.data.utils import pad_list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, CropForegroundd, DivisiblePadd, RandAffined
from monai.utils import set_determinism

from monailabel.tasks.scoring.tta import TestTimeAugmentation

trange = partial(tqdm.trange, desc="training")


class TestTestTimeAugmentation(unittest.TestCase):
    @staticmethod
    def get_data(num_examples, input_size, data_type=np.asarray, include_label=True):
        custom_create_test_image_2d = partial(
            create_test_image_2d, *input_size, rad_max=7, num_seg_classes=1, num_objs=1
        )
        data = []
        for _ in range(num_examples):
            im, label = custom_create_test_image_2d()
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

    def test_test_time_augmentation(self):
        input_size = (20, 20)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        keys = ["image", "label"]
        num_training_ims = 10
        train_data = self.get_data(num_training_ims, input_size)
        test_data = self.get_data(1, input_size)

        transforms = Compose(
            [
                AddChanneld(keys),
                RandAffined(
                    keys,
                    prob=1.0,
                    spatial_size=(30, 30),
                    rotate_range=(np.pi / 3, np.pi / 3),
                    translate_range=(3, 3),
                    scale_range=((0.8, 1), (0.8, 1)),
                    padding_mode="zeros",
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                CropForegroundd(keys, source_key="image"),
                DivisiblePadd(keys, 4),
            ]
        )

        train_ds = CacheDataset(train_data, transforms)
        # output might be different size, so pad so that they match
        train_loader = DataLoader(train_ds, batch_size=2, collate_fn=pad_list_data_collate)

        model = UNet(2, 1, 1, channels=(6, 6), strides=(2, 2)).to(device)
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

        post_trans = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold_values=True),
            ]
        )

        def inferrer_fn(x):
            return post_trans(model(x))

        tt_aug = TestTimeAugmentation(transforms, batch_size=5, num_workers=0, inferrer_fn=inferrer_fn, device=device)
        mode, mean, std, vvc = tt_aug(test_data)
        self.assertEqual(mode.shape, (1,) + input_size)
        self.assertEqual(mean.shape, (1,) + input_size)
        self.assertTrue(all(np.unique(mode) == (0, 1)))
        self.assertEqual((mean.min(), mean.max()), (0.0, 1.0))
        self.assertEqual(std.shape, (1,) + input_size)
        self.assertIsInstance(vvc, float)


if __name__ == "__main__":
    unittest.main()
