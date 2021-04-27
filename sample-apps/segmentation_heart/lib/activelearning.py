'''
Consider implementing a light version for TTA specified in this paper: https://arxiv.org/pdf/2007.00833.pdf
'''
import os
import numpy as np
import torch
from glob import glob
from functools import partial

from monailabel.interface import ActiveLearning

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

device = torch.device("cuda:0")

def tta(root_dir, task, bestModelPath, num_examples):

    # Using unet
    model = UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.2).to(device)

    # Performing inference on the dataloader/image
    model.load_state_dict(torch.load(bestModelPath))
    model.eval()

    ######### REMEMBER THAT THESE TRANSFORMS SHOULD KEEP THE AREA TO SEGMENT IN THE FIELD OF VIEW!! ##############
    transforms = Compose([
                        LoadImaged(keys=["image"]),
                        EnsureChannelFirstd(keys=["image"]),
                        RandAffined(keys=["image"], prob=1,
                                    rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                                    padding_mode="zeros",
                                    as_tensor_output=False),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        ToTensord(keys=["image"]),
                    ])

    # Load test images
    all_imgs = sorted(glob(os.path.join(root_dir, task, 'imagesTs', '*nii.gz')))
    data_dicts = [{"image": image} for image in all_imgs]
    ds_tta = monai.data.Dataset(data=data_dicts)
    loader_tta = DataLoader(ds_tta, batch_size=1, num_workers=0, collate_fn=list_data_collate)

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
        device=device
    )

    vvc_tta_all = []
    for idx, file in enumerate(loader_tta):
        print(f'Processing image: {idx+1}')
        mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(file, num_examples=num_examples)
        vvc_tta_all.append(vvc_tta)
        print('Volume Variation Coefficient: ', vvc_tta)
