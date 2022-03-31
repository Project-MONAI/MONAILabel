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
import copy
import logging
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import list_data_collate, pad_list_data_collate
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandRotated,
    Resized,
    ToTensord,
)
from monai.transforms.compose import Compose
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.inverse_batch_transform import BatchInverseTransform
from monai.transforms.transform import Randomizable
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils.enums import CommonKeys, InverseKeys
from tqdm import tqdm

from monailabel.deepedit.transforms import DiscardAddGuidanced
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.scoring import ScoringMethod

logger = logging.getLogger(__name__)


class TTAScoring(ScoringMethod):
    """
    First version of test time augmentation active learning
    """

    def __init__(
        self, model, network=None, deepedit=True, num_samples=5, spatial_size=None, spacing=None, load_strict=False
    ):
        super().__init__("Compute initial score based on TTA")
        if spacing is None:
            spacing = [1.0, 1.0, 1.0]
        if spatial_size is None:
            spatial_size = [128, 128, 128]
        self.model = model
        self.device = "cuda"
        self.num_samples = num_samples
        self.network = network
        self.deepedit = deepedit
        self.spatial_size = spatial_size
        self.spacing = spacing
        self.load_strict = load_strict

    def pre_transforms(self):
        t = [
            LoadImaged(keys="image", reader="nibabelreader"),
            AddChanneld(keys="image"),
            # Spacing might not be needed as resize transform is used later.
            # Spacingd(keys="image", pixdim=self.spacing),
            RandAffined(
                keys="image",
                prob=1,
                rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                padding_mode="zeros",
                as_tensor_output=False,
            ),
            RandFlipd(keys="image", prob=0.5, spatial_axis=0),
            RandRotated(keys="image", range_x=(-5, 5), range_y=(-5, 5), range_z=(-5, 5)),
            Resized(keys="image", spatial_size=self.spatial_size),
        ]
        # If using TTA for deepedit
        if self.deepedit:
            t.append(DiscardAddGuidanced(keys="image"))
        t.append(ToTensord(keys="image"))
        return Compose(t)

    def post_transforms(self):
        return Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold_values=True),
            ]
        )

    def _inferer(self, images, model):
        preds = SimpleInferer()(images, model)
        transforms = self.post_transforms()
        post_pred = transforms(preds)
        return post_pred

    @staticmethod
    def _get_model_path(path):
        if not path:
            return None

        paths = [path] if isinstance(path, str) else path
        for path in reversed(paths):
            if os.path.exists(path):
                return path
        return None

    def _load_model(self, path, network):
        model_file = TTAScoring._get_model_path(path)
        if not model_file and not network:
            logger.warning(f"Skip TTA Scoring:: Model(s) {path} not available yet")
            return None, None

        logger.info(f"Using {model_file} for running TTA")
        model_ts = int(os.stat(model_file).st_mtime) if model_file and os.path.exists(model_file) else 1
        if network:
            model = copy.deepcopy(network)
            if model_file:
                checkpoint = torch.load(model_file)
                model_state_dict = checkpoint.get("model", checkpoint)
                model.load_state_dict(model_state_dict, strict=self.load_strict)
        else:
            model = torch.jit.load(model_file)
        return model, model_ts

    def __call__(self, request, datastore: Datastore):
        logger.info("Starting TTA scoring")

        result = {}
        model, model_ts = self._load_model(self.model, self.network)
        if not model:
            return
        model = model.to(self.device).eval()

        tt_aug = TestTimeAugmentation(
            transform=self.pre_transforms(),
            label_key="image",
            batch_size=1,
            num_workers=0,
            inferrer_fn=partial(self._inferer, model=model),
            device=self.device,
            progress=self.num_samples > 1,
        )

        # Performing TTA for all unlabeled images
        skipped = 0
        unlabeled_images = datastore.get_unlabeled_images()
        num_samples = request.get("num_samples", self.num_samples)

        logger.info(f"TTA:: Total unlabeled images: {len(unlabeled_images)}")
        for image_id in unlabeled_images:
            image_info = datastore.get_image_info(image_id)
            prev_ts = image_info.get("tta_ts", 0)
            if prev_ts == model_ts:
                skipped += 1
                continue

            logger.info(f"TTA:: Run for image: {image_id}; Prev Ts: {prev_ts}; New Ts: {model_ts}")

            # Computing the Volume Variation Coefficient (VVC)
            start = time.time()
            with torch.no_grad():
                data = {"image": datastore.get_image_uri(image_id)}
                tta_mode, tta_mean, tta_std, tta_vvc = tt_aug(data, num_examples=num_samples)

            logger.info(f"TTA:: {image_id} => vvc: {tta_vvc}")
            if self.device == "cuda":
                torch.cuda.empty_cache()

            latency_tta = time.time() - start
            logger.info(f"TTA:: Time taken for {num_samples} augmented samples: {latency_tta} (sec)")

            # Add vvc in datastore
            info = {"tta_vvc": tta_vvc, "tta_ts": model_ts}
            datastore.update_image_info(image_id, info)
            result[image_id] = info

        logger.info(f"TTA:: Total: {len(unlabeled_images)}; Skipped = {skipped}; Executed: {len(result)}")
        return result


class TestTimeAugmentation:
    """
    Class for performing test time augmentations. This will pass the same image through the network multiple times.

    The user passes transform(s) to be applied to each realisation, and provided that at least one of those transforms
    is random, the network's output will vary. Provided that inverse transformations exist for all supplied spatial
    transforms, the inverse can be applied to each realisation of the network's output. Once in the same spatial
    reference, the results can then be combined and metrics computed.

    Test time augmentations are a useful feature for computing network uncertainty, as well as observing the network's
    dependency on the applied random transforms.

    Reference:
        Wang et al.,
        Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional
        neural networks,
        https://doi.org/10.1016/j.neucom.2019.01.103

    Args:
        transform: transform (or composed) to be applied to each realisation. At least one transform must be of type
            `Randomizable`. All random transforms must be of type `InvertibleTransform`.
        batch_size: number of realisations to infer at once.
        num_workers: how many subprocesses to use for data.
        inferrer_fn: function to use to perform inference.
        device: device on which to perform inference.
        image_key: key used to extract image from input dictionary.
        label_key: key used to extract label from input dictionary.
        meta_keys: explicitly indicate the key of the expected meta data dictionary.
            for example, for data with key `label`, the metadata by default is in `label_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
            this arg only works when `meta_keys=None`.
        return_full_data: normally, metrics are returned (mode, mean, std, vvc). Setting this flag to `True` will return the
            full data. Dimensions will be same size as when passing a single image through `inferrer_fn`, with a dimension appended
            equal in size to `num_examples` (N), i.e., `[N,C,H,W,[D]]`.
        progress: whether to display a progress bar.

    Example:
        .. code-block:: python

            transform = RandAffined(keys, ...)
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

            tt_aug = TestTimeAugmentation(
                transform, batch_size=5, num_workers=0, inferrer_fn=lambda x: post_trans(model(x)), device=device
            )
            mode, mean, std, vvc = tt_aug(test_data)
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        batch_size: int,
        num_workers: int,
        inferrer_fn: Callable,
        device: Union[str, torch.device] = "gpu",
        image_key=CommonKeys.IMAGE,
        label_key=CommonKeys.LABEL,
        meta_keys: Optional[str] = None,
        meta_key_postfix="meta_dict",
        return_full_data: bool = False,
        progress: bool = True,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inferrer_fn = inferrer_fn
        self.device = device
        self.image_key = image_key
        self.label_key = label_key
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix
        self.return_full_data = return_full_data
        self.progress = progress

        # check that the transform has at least one random component, and that all random transforms are invertible
        self._check_transforms()

    def _check_transforms(self):
        """Should be at least 1 random transform, and all random transforms should be invertible."""
        ts = [self.transform] if not isinstance(self.transform, Compose) else self.transform.transforms
        randoms = np.array([isinstance(t, Randomizable) for t in ts])
        invertibles = np.array([isinstance(t, InvertibleTransform) for t in ts])
        # check at least 1 random
        if sum(randoms) == 0:
            raise RuntimeError(
                "Requires a `Randomizable` transform or a `Compose` containing at least one `Randomizable` transform."
            )
        # check that whenever randoms is True, invertibles is also true
        for r, i in zip(randoms, invertibles):
            if r and not i:
                raise RuntimeError(
                    f"All applied random transform(s) must be invertible. Problematic transform: {type(r).__name__}"
                )

    def __call__(
        self, data: Dict[str, Any], num_examples: int = 10
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]:
        """
        Args:
            data: dictionary data to be processed.
            num_examples: number of realisations to be processed and results combined.

        Returns:
            - if `return_full_data==False`: mode, mean, std, vvc. The mode, mean and standard deviation are calculated across
                `num_examples` outputs at each voxel. The volume variation coefficient (VVC) is `std/mean` across the whole output,
                including `num_examples`. See original paper for clarification.
            - if `return_full_data==False`: data is returned as-is after applying the `inferrer_fn` and then concatenating across
                the first dimension containing `num_examples`. This allows the user to perform their own analysis if desired.
        """
        d = dict(data)

        # check num examples is multiple of batch size
        if num_examples % self.batch_size != 0:
            raise ValueError("num_examples should be multiple of batch size.")

        # generate batch of data of size == batch_size, dataset and dataloader
        data_in = [d] * num_examples
        ds = Dataset(data_in, self.transform)
        dl = DataLoader(ds, self.num_workers, batch_size=self.batch_size, collate_fn=pad_list_data_collate)

        label_transform_key = self.label_key + InverseKeys.KEY_SUFFIX

        # create inverter
        inverter = BatchInverseTransform(self.transform, dl, collate_fn=list_data_collate)

        outputs: List[np.ndarray] = []

        for batch_data in tqdm(dl) if self.progress else dl:
            batch_images = batch_data[self.image_key].to(self.device)

            # do model forward pass
            batch_output = self.inferrer_fn(batch_images)
            if isinstance(batch_output, torch.Tensor):
                batch_output = batch_output.detach().cpu()
            if isinstance(batch_output, np.ndarray):
                batch_output = torch.Tensor(batch_output)

            # create a dictionary containing the inferred batch and their transforms
            inferred_dict = {self.label_key: batch_output, label_transform_key: batch_data[label_transform_key]}
            # if meta dict is present, add that too (required for some inverse transforms)
            label_meta_dict_key = self.meta_keys or f"{self.label_key}_{self.meta_key_postfix}"
            if label_meta_dict_key in batch_data:
                inferred_dict[label_meta_dict_key] = batch_data[label_meta_dict_key]

            # do inverse transformation (allow missing keys as only inverting label)
            with allow_missing_keys_mode(self.transform):  # type: ignore
                inv_batch = inverter(inferred_dict)

            # append
            outputs.append(inv_batch[self.label_key])

        # output
        output: np.ndarray = np.concatenate(outputs)

        if self.return_full_data:
            return output

        # calculate metrics
        mode = np.array(torch.mode(torch.Tensor(output.astype(np.int64)), dim=0).values)
        mean: np.ndarray = np.mean(output, axis=0)  # type: ignore
        std: np.ndarray = np.std(output, axis=0)  # type: ignore
        vvc: float = (np.std(output) / np.mean(output)).item()
        return mode, mean, std, vvc
