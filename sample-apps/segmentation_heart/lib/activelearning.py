'''
Consider implementing a light version of TTA presented in this paper: https://arxiv.org/pdf/2007.00833.pdf
'''
import numpy as np
import torch
from functools import partial

from monai.networks.nets import UNet
import monai
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import (
    list_data_collate,
    DataLoader,
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


class MyActiveLearning():
    def __init__(self, bestModelPath):

        self.bestModelPath = bestModelPath
        self.device = torch.device("cuda:0")
        self.num_examples = 2 # Number of augmented samples

    def get_model(self):

        # Using UNet
        # For most of the Active Learning techniques, this model is the same one we used for inference
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2).to(self.device)

        model.load_state_dict(torch.load(self.bestModelPath))
        model.eval()

        return model

    def __call__(self, images):

        # Creating dataloader
        data_dicts = [{"image": image} for image in images]
        ds_tta = monai.data.Dataset(data=data_dicts)
        loader_tta = DataLoader(ds_tta, batch_size=1, num_workers=0, collate_fn=list_data_collate)

        model = self.get_model()

        # Defining transforms
        transforms = Compose([
          LoadImaged(keys=["image"]),
          EnsureChannelFirstd(keys=["image"]),
          RandAffined(keys=["image"], prob=1,
                      rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                      padding_mode="zeros",
                      as_tensor_output=False),
          NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
          ToTensord(keys=["image"]),
        ])

        ##### Performing TTA ######

        post_trans = Compose([
          Activations(sigmoid=True),
          AsDiscrete(threshold_values=True),
        ])

        # Inferer function used in the TTA
        def infer_seg(images, model, roi_size=(160, 192, 80), sw_batch_size=1):
          preds = sliding_window_inference(images, roi_size, sw_batch_size, model)
          post_pred = post_trans(preds)
          return post_pred

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

        # Returning image with higher VVC (Volume Variation Coefficient)
        return {'image': images[np.array(vvc_tta_all).argmax()]}