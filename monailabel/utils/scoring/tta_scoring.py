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

import logging
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandRotated,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks import ScoringMethod
from monailabel.utils.scoring.test_time_augmentation import TestTimeAugmentation

logger = logging.getLogger(__name__)


class TTAScoring(ScoringMethod):
    """
    First version of test time augmentation active learning
    """

    def __init__(self, model, network=None, plot=None):
        super().__init__("Compute initial score based on TTA")
        self.model = model
        self.device = "cuda"
        self.img_size = [128, 128, 128]
        self.num_samples = 5
        self.network = network
        self.plot = plot

    def pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys="image"),
                AddChanneld(keys="image"),
                Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0]),
                RandAffined(
                    keys="image",
                    prob=1,
                    rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                    padding_mode="zeros",
                    as_tensor_output=False,
                ),
                RandFlipd(keys="image", prob=0.5, spatial_axis=0),
                RandRotated(keys="image", range_x=(-5, 5), range_y=(-5, 5), range_z=(-5, 5)),
                Resized(keys="image", spatial_size=self.img_size),
                ToTensord(keys="image"),
            ]
        )

    def post_transforms(self):
        return Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold_values=True),
            ]
        )

    def infer_seg(self, images, model, roi_size, sw_batch_size):
        # preds = sliding_window_inference(images, roi_size, sw_batch_size, model)
        preds = SimpleInferer()(images, model)
        transforms = self.post_transforms()
        post_pred = transforms(preds)
        return post_pred

    def get_2d_im(self, im, channel, z_slice):
        im = im[..., z_slice]
        if channel is not None:
            im = im[channel][None]
        return im

    def imshows(self, ims):
        nrow = len(ims)
        ncol = len(ims[0])
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
        for i, im_dict in enumerate(ims):
            for j, (title, im) in enumerate(im_dict.items()):
                if isinstance(im, torch.Tensor):
                    im = im.detach().cpu().numpy()
                im = np.mean(im, axis=0)  # average across channels
                ax = axes[j] if len(ims) == 1 else axes[i, j]
                ax.set_title(f"{title}\n{im.shape}")
                im_show = ax.imshow(im)
                ax.axis("off")
                fig.colorbar(im_show, ax=ax)
                fig.savefig("/home/adp20local/Documents/MONAILabel/sample-apps/segmentation_spleen_tta/tta_output.png")

    def get_model_path(self, path):
        if not path:
            return None

        paths = [path] if isinstance(path, str) else path
        for path in reversed(paths):
            if os.path.exists(path):
                return path
        return None

    def __call__(self, request, datastore: Datastore):

        logger.info("Starting TTA scoring")

        result = {}

        path = self.get_model_path(self.model)

        if self.network:
            model = self.network
            if path:
                checkpoint = torch.load(path)
                model_state_dict = checkpoint.get("model", checkpoint)
                model.load_state_dict(model_state_dict)
        else:
            model = torch.jit.load(path)

        tt_aug = TestTimeAugmentation(
            transform=self.pre_transforms(),
            label_key="image",
            batch_size=1,
            num_workers=0,
            inferrer_fn=partial(self.infer_seg, roi_size=self.img_size, model=model.to(self.device), sw_batch_size=1),
            device=self.device,
        )

        to_imshow = []
        idx = 0
        # Performing TTA for all unlabeled images
        for image_id in datastore.get_unlabeled_images():

            logger.info("TTA for image: " + image_id)

            file = {"image": datastore.get_image_uri(image_id)}

            # Computing the Volume Variation Coefficient (VVC)
            start = time.time()
            with torch.no_grad():
                mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=self.num_samples)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            latency_tta = time.time() - start

            logger.info("Time taken for " + str(self.num_samples) + " augmented samples: " + str(latency_tta))

            # Add vvc in datastore
            info = {"vvc_tta": vvc_tta}
            logger.info(f"{image_id} => {info}")
            datastore.update_image_info(image_id, info)
            result[image_id] = info

            if self.plot:
                im_gt = LoadImaged(keys="image")(file)
                im_gt = im_gt["image"][None]
                # Preparing images to plot
                to_imshow.append(
                    {
                        "im GT": self.get_2d_im(im_gt, None, int(im_gt.shape[-1] / 1.5)),
                        # "label GT": self.get_2d_im(label_gt, None, int(label_gt.shape[-1]/3)),
                        "mode, vvc: %.2f" % vvc_tta: self.get_2d_im(mode_tta, None, int(mode_tta.shape[-1] / 1.5)),
                        "mean, vvc: %.2f" % vvc_tta: self.get_2d_im(mean_tta, None, int(mean_tta.shape[-1] / 1.5)),
                        "std, vvc: %.2f" % vvc_tta: self.get_2d_im(std_tta, None, int(std_tta.shape[-1] / 1.5)),
                    }
                )
            idx += 1
            if idx > 0:
                # Plotting images
                self.imshows(to_imshow)

        return result
