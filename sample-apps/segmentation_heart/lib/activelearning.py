'''
Consider implementing a light version for TTA specified in this paper: https://arxiv.org/pdf/2007.00833.pdf
'''
import os
import numpy as np
import torch
from glob import glob
from functools import partial

from monailabel.interface import ActiveLearning
MyActiveLearning = ActiveLearning

from monai.networks.nets import UNet
import monai
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import (
    list_data_collate,
    DataLoader,
    Dataset,
    TestTimeAugmentation,
)
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    ToTensord,
)


class MyActiveLearning(ActiveLearning):
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.num_examples = num_examples

    def get_model(self, bestModelPath):
        # Using unet
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2).to(self.device)
        # Performing inference on the dataloader/image
        model.load_state_dict(torch.load(bestModelPath))
        model.eval()

        return model

    def __call__(self, request):

        model = self.get_model(request.pretrainedModel)

        transforms = Compose([
          LoadImaged(keys=["image"]),
          EnsureChannelFirstd(keys=["image"]),
          RandAffined(keys=["image"], prob=1,
                      rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                      padding_mode="zeros",
                      as_tensor_output=False),
          NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
          ToTensord(keys=["image"]),
        ])

        # Performing TTA
        post_trans = Compose([
          Activations(sigmoid=True),
          AsDiscrete(threshold_values=True),
        ])

        # Function used in the TTA
        def infer_seg(images, model, roi_size=(160, 192, 80), sw_batch_size=1):
          preds = sliding_window_inference(images, roi_size, sw_batch_size, model)
          post_pred = post_trans(preds)
          return post_pred

        # TTA
        tt_aug = TestTimeAugmentation(
          transforms,
          label_key="image",
          batch_size=1,
          num_workers=0,
          inferrer_fn=partial(infer_seg, model=model),
          device=self.device
        )

        vvc_tta_all = []
        for idx, file in enumerate(loader_tta):
            print(f'Processing image: {idx + 1}')
            mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=self.num_examples)
            vvc_tta_all.append(vvc_tta)
            print('Volume Variation Coefficient: ', vvc_tta)

        # Test images are sorted according to the volume variation coefficient (vvc_tta_all).
        # The bigger the vvc, the more "uncertain" is the segmentation.

        return image