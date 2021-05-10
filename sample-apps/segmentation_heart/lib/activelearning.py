import logging
import os
from functools import partial

import monai
import numpy as np
import torch
from monai.data import DataLoader, TestTimeAugmentation, list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    ToTensord,
)

from monailabel.interfaces import ActiveLearning, Datastore
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException

logger = logging.getLogger(__name__)


class MyActiveLearning(ActiveLearning):
    """
    Consider implementing a light version of TTA presented in this paper: https://arxiv.org/pdf/2007.00833.pdf
    """

    def __init__(self, path, network=None):
        self.path = path
        self.network = network
        self.num_examples = 2  # Number of augmented samples
        self.max_images = 3

        # Defining transforms
        self.pre_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                RandAffined(
                    keys=["image"],
                    prob=1,
                    rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
                    padding_mode="zeros",
                    as_tensor_output=False,
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ToTensord(keys=["image"]),
            ]
        )

        self.post_transforms = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold_values=True),
            ]
        )

    def __call__(self, request, datastore: Datastore):
        return super().__call__(request, datastore)

    def get_model(self, device):
        if not os.path.exists(os.path.join(self.path)):
            raise MONAILabelException(
                MONAILabelError.MODEL_IMPORT_ERROR,
                f"Model Path ({self.path}) does not exist",
            )

        if self.network:
            network = self.network
            network.load_state_dict(torch.load(self.path))
        else:
            network = torch.jit.load(self.path)

        network = network.to(device) if device else network
        network.eval()
        return network

    def __call__(self, request, datastore: Datastore):
        if request.get("strategy") != "tta":
            return super().__call__(request, datastore)

        images = datastore.get_unlabeled_images()
        images = images[: self.max_images]
        logger.info(f"Total Unlabeled Images: {len(images)}")

        # Creating dataloader
        data_dicts = [{"image": image} for image in images]
        ds_tta = monai.data.Dataset(data=data_dicts)
        loader_tta = DataLoader(
            ds_tta, batch_size=1, num_workers=0, collate_fn=list_data_collate
        )

        device = torch.device(request.get("device", "cuda:0"))
        logger.info(f"Using device: {device}")
        model = self.get_model(device)

        # Performing TTA
        # Inferer function used in the TTA
        def infer_seg(images, model, roi_size=(160, 192, 80), sw_batch_size=1):
            preds = sliding_window_inference(images, roi_size, sw_batch_size, model)
            post_pred = self.post_transforms(preds)
            return post_pred

        tt_aug = TestTimeAugmentation(
            self.pre_transforms,
            label_key="image",
            batch_size=1,
            num_workers=0,
            inferrer_fn=partial(infer_seg, model=model),
            device=device,
        )

        vvc_tta_all = []
        for idx, file in enumerate(loader_tta):
            logger.info(f"Processing image: {idx + 1}")
            mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(
                file, num_examples=self.num_examples
            )
            vvc_tta_all.append(vvc_tta)
            logger.info(f"Volume Variation Coefficient: {vvc_tta}")

        # Returning image with higher VVC (Volume Variation Coefficient)
        image = images[np.array(vvc_tta_all).argmax()]
        logger.info(f"Strategy: tta; Selected Image: {image}")
        return image
