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
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

# from monai.data import TestTimeAugmentation
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks import ScoringMethod
from monailabel.utils.scoring.test_time_augmentation import TestTimeAugmentation

logger = logging.getLogger(__name__)


class EpistemicScoring(ScoringMethod):
    """
    First version of test time augmentation active learning
    """

    def __init__(self, model=None, plot=False):
        super().__init__("Compute initial score based on TTA")
        self.model = model
        self.device = "cuda"
        self.img_size = [128, 128, 64]
        self.num_samples = 10
        self.plot = plot

    def pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys="image"),
                AddChanneld(keys="image"),
                Spacingd(keys="image", pixdim=[1.5, 1.5, 2.0]),
                ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=("image"), source_key="image"),
                ToTensord(keys="image"),
            ]
        )

    def post_transforms(self):
        return Compose(
            [
                Activations(softmax=True),
                #AsDiscrete(threshold_values=True),
            ]
        )

    def infer_seg(self, images, model, roi_size, sw_batch_size):
        pre_transforms = self.pre_transforms()
        images = pre_transforms(images)
        preds = sliding_window_inference(images, roi_size, sw_batch_size, model)
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

    def entropy_3d_volume(vol_input):
        # The input is assumed with repetitions, channels and then volumetric data
        vol_input = vol_input.astype(dtype='float32')
        dims = vol_input.shape
        reps = dims[0]
        entropy = np.zeros(dims[2:], dtype='float32')

        # Threshold values less than or equal to zero
        threshold = 0.00005
        vol_input[vol_input <= 0] = threshold

        # Looping across channels as each channel is a class
        if len(dims) == 5:
            for channel in range(dims[1]):
                t_vol = np.squeeze(vol_input[:, channel, :, :, :])
                t_sum = np.sum(t_vol, axis=0)
                t_avg = np.divide(t_sum, reps)
                t_log = np.log(t_avg)
                t_entropy = -np.multiply(t_avg, t_log)
                entropy = entropy + t_entropy
        else:
            t_vol = np.squeeze(vol_input)
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy

        return entropy

    def __call__(self, request, datastore: Datastore):

        logger.info("Starting Epistemic Uncertainty scoring")

        result = {}

        '''
        tt_aug = TestTimeAugmentation(
            transform=self.pre_transforms(),
            label_key="image",
            batch_size=1,
            num_workers=0,
            inferrer_fn=partial(
                self.infer_seg, roi_size=self.img_size, model=self.model._get_network(self.device), sw_batch_size=1
            ),
            device=self.device,
        )
        '''

        to_imshow = []
        # Performing TTA for all unlabeled images
        for image_id in datastore.get_unlabeled_images():

            logger.info("Epistemic Uncertainty for image: " + image_id)

            file = {"image": datastore.get_image_uri(image_id)}

            # Computing the Volume Variation Coefficient (VVC)
            start = time.time()
            with torch.no_grad():

                accum_unl_outputs = []
                for mc in range(self.num_samples):
                    output_pred = self.infer_seg(images=file,
                                                 model=self.model._get_network(self.device),
                                                 sw_batch_size=1,
                                                 roi_size=self.img_size)

                    # Accumulate
                    accum_unl_outputs.append(output_pred)

                # Stack it up
                accum_tensor = torch.stack(accum_unl_outputs)
                # Squeeze
                accum_tensor = torch.squeeze(accum_tensor)
                # Send to CPU
                accum_numpy = accum_tensor.to('cpu').numpy()
                accum_numpy = accum_numpy[:, 1, :, :, :]

                entropy = self.entropy_3d_volume(accum_numpy)
                entropy_sum = np.sum(entropy)

                #mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=self.num_samples)

            if self.device == "cuda":
                torch.cuda.empty_cache()
            latency_tta = time.time() - start

            logger.info("Time taken for " + str(self.num_samples) + " Monte Carlo Simulation samples: " + str(latency_tta))

            # Add vvc in datastore
            info = {"entropy_sum": entropy_sum}
            logger.info(f"{image_id} => {info}")
            datastore.update_image_info(image_id, info)
            result[image_id] = info

            '''
            if self.plot:
                im_gt = LoadImaged(keys="image")(file)
                im_gt = im_gt["image"][None]

                # Preparing images to plot
                to_imshow.append(
                    {
                        "im GT": self.get_2d_im(im_gt, None, int(im_gt.shape[-1] / 1.5)),
                        # "label GT": self.get_2d_im(label_gt, None, int(label_gt.shape[-1]/3)),
                        "mode, vvc: %.2f" % vvc_tta: self.get_2d_im(mode_tta, 1, int(mode_tta.shape[-1] / 1.5)),
                        "mean, vvc: %.2f" % vvc_tta: self.get_2d_im(mean_tta, 1, int(mean_tta.shape[-1] / 1.5)),
                        "std, vvc: %.2f" % vvc_tta: self.get_2d_im(std_tta, 1, int(std_tta.shape[-1] / 1.5)),
                    }
                )
                # Plotting images
                self.imshows(to_imshow)
            '''

        return result
