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
import copy
import logging
from typing import Any, Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms import CropForeground, GaussianSmooth, Randomizable, Resize, ScaleIntensity, SpatialCrop
from monai.transforms.transform import MapTransform, Transform
from monai.utils.enums import CommonKeys

LABELS_KEY = "label_names"

logger = logging.getLogger(__name__)


class BinaryMaskd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Convert to single label - This should actually create the heat map for the first stage

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[key][d[key] > 0] = 1
        return d


class SelectVertebraAndCroppingd(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Crop image and label

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Logic:

            # 1/ randomly select vertebra
            # 2/ binarise volume
            # 3/ apply cropForeground transform

            # Alternative Logic:

            # 1/ randomly select vertebra
            # 2/ spatial crop based on the centroid
            # 3/ binarise cropped volume

            d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]
            tmp_label = copy.deepcopy(d[key])

            # TO DO: WHAT'S THE BEST WAY TO SELECT A DIFFERENT SEGMENT EACH ITERATION - Randomizable should work?
            # PERHAPS DOING BATCHES AS IN TRANSFORM RandCropByLabelClassesd??

            current_idx = self.R.randint(0, len(d["centroids"]))

            d["current_idx"] = current_idx

            d["current_label"] = list(d["centroids"][current_idx].values())[0][0]

            logger.info(f'Processing vertebra: {d["current_label"]}')

            # Make binary the label
            tmp_label[tmp_label != d["current_label"]] = 0
            tmp_label[tmp_label > 0] = 1

            ##########
            # Cropping
            ##########
            def condition(x):
                # threshold at 1
                return x > 0

            cropper = CropForeground(select_fn=condition, margin=4)

            start, stop = cropper.compute_bounding_box(tmp_label)

            slices_cropped = [[start[-3], stop[-3]], [start[-2], stop[-2]], [start[-1], stop[-1]]]

            d["slices_cropped"] = slices_cropped

            # Cropping label
            d["label"] = tmp_label[:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

            # Cropping image
            d["image"] = d["image"][:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

        return d


class GetCentroidsd(MapTransform):
    def __init__(self, keys: KeysCollection, centroids_key: str = "centroids", allow_missing_keys: bool = False):
        """
        Get centroids

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.centroids_key = centroids_key

    def _get_centroids(self, label):
        centroids = []
        # loop over all segments
        areas = []
        for seg_class in np.unique(label):
            c = {}
            # skip background
            if seg_class == 0:
                continue
            # get centre of mass (CoM)
            centre = []
            for indices in np.where(label == seg_class):
                avg_indices = np.average(indices).astype(int)
                centre.append(avg_indices)
            c[f"label_{int(seg_class)}"] = [int(seg_class), centre[-3], centre[-2], centre[-1]]
            centroids.append(c)
        return centroids

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Get centroids
            d[self.centroids_key] = self._get_centroids(d[key])
        return d


class GaussianSmoothedCentroidd(MapTransform):
    def __init__(self, keys: KeysCollection, signal_key: str = "signal", allow_missing_keys: bool = False):
        """
        Apply Gaussian to Centroid

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.signal_key = signal_key

    def __call__(self, data):
        d: Dict = dict(data)

        logger.info("Processing label: " + d["label_meta_dict"]["filename_or_obj"])

        signal = np.zeros((1, d["original_size"][-3], d["original_size"][-2], d["original_size"][-1]), dtype=np.float32)

        x, y, z = (
            list(d["centroids"][d["current_idx"]].values())[0][-3],
            list(d["centroids"][d["current_idx"]].values())[0][-2],
            list(d["centroids"][d["current_idx"]].values())[0][-1],
        )
        signal[:, x, y, z] = 1.0

        signal = signal[
            :,
            d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
            d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
            d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
        ]

        sigma = 1.6 + (d["current_label"] - 1.0) * 0.1

        signal = GaussianSmooth(sigma)(signal)

        # d[self.signal_key] = signal * d["label"] # use signal only inside mask?
        d[self.signal_key] = signal

        return d


class ConcatenateROId(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Add Gaussian smoothed centroid (signal) to cropped volume

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            tmp_image = np.concatenate([d["image"], d[key]], axis=0)
            d["image"] = tmp_image
        return d


class PlaceCroppedAread(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Place the ROI predicted in the full image

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for _ in self.key_iterator(d):
            final_pred = np.zeros(
                (1, d["original_size"][-3], d["original_size"][-2], d["original_size"][-1]), dtype=np.float32
            )
            #  Undo/invert the resize of d["pred"] #
            d["pred"] = Resize(spatial_size=d["cropped_size"], mode="nearest")(d["pred"])
            final_pred[
                :,
                d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
                d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
                d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
            ] = d["pred"]
            d["pred"] = final_pred * int(d["current_label"])
        return d


# For the second stage - Vertebra localization


class VertHeatMap(MapTransform):
    def __init__(self, keys, gamma=1000.0, label_names=None):
        super().__init__(keys)
        self.label_names = label_names
        self.gamma = gamma

    def __call__(self, data):
        for k in self.keys:
            i = data[k].long()
            # one hot if necessary
            is_onehot = i.shape[0] > 1
            if is_onehot:
                out = torch.zeros_like(i)
            else:
                out = torch.nn.functional.one_hot(i, len(self.label_names) + 1)  # plus background
                out = torch.movedim(out[0], -1, 0)
                out.fill_(0.0)
                out = out.float()

            # loop over all segmentation classes
            for seg_class in torch.unique(i):
                # skip background
                if seg_class == 0:
                    continue
                # get CoM for given segmentation class
                centre = [np.average(indices.cpu()).astype(int) for indices in torch.where(i[0] == seg_class)]
                label_num = seg_class.item()
                centre.insert(0, label_num)
                out[tuple(centre)] = 1.0
                sigma = 1.6 + (label_num - 1.0) * 0.1
                # Gaussian smooth
                out[label_num] = GaussianSmooth(sigma)(out[label_num].cuda()).cpu()
                # Normalize to [0,1]
                out[label_num] = ScaleIntensity()(out[label_num])
                out[label_num] = out[label_num] * self.gamma

            # TO DO: Keep the centroids in the data dictionary?

            data[k] = out

        return data


class VertebraLocalizationPostProcessing(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", allow_missing_keys: bool = False):
        """
        Postprocess Vertebra localization

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.result = result

    def __call__(self, data):
        d: Dict = dict(data)
        centroids = []
        for key in self.key_iterator(d):
            # Getting centroids
            for l in range(d[key].shape[0] - 1):
                centroid = {}
                if d[key][l + 1, ...].max() < 30.0:
                    continue
                x, y, z = np.where(d[key][l + 1, ...] == d[key][l + 1, ...].max())
                x, y, z = x[0], y[0], z[0]
                centroid[f"label_{l + 1}"] = [x, y, z]
                centroids.append(centroid)

            print(centroids)
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result]["centroids"] = centroids
        return d


class VertebraLocalizationSegmentation(MapTransform):
    def __init__(self, keys: KeysCollection, result: str = "result", allow_missing_keys: bool = False):
        """
        Postprocess Vertebra localization using segmentation task

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.result = result

    def _get_centroids(self, label):
        centroids = []
        # loop over all segments
        areas = []
        for seg_class in torch.unique(label):
            c = {}
            # skip background
            if seg_class == 0:
                continue

            # get centre of mass (CoM)
            centre = []
            for indices in torch.where(label == seg_class):
                # most_indices = np.percentile(indices, 60).astype(int).tolist()
                # centre.append(most_indices)
                avg_indices = np.average(indices).astype(int).tolist()
                centre.append(avg_indices)

            if len(indices) < 1000:
                continue

            areas.append(len(indices))
            c[f"label_{int(seg_class)}"] = [int(seg_class), centre[-3], centre[-2], centre[-1]]
            centroids.append(c)

        # Rules to discard centroids
        # 1/ Should we consider the distance between centroids?
        # 2/ Should we consider the area covered by the vertebra

        return centroids

    def __call__(self, data):
        d: Dict = dict(data)
        centroids = []
        for key in self.key_iterator(d):
            # Getting centroids
            centroids = self._get_centroids(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result]["centroids"] = centroids
        return d


class CropAndCreateSignald(MapTransform):
    def __init__(self, keys: KeysCollection, signal_key, allow_missing_keys: bool = False):
        """
        Based on the centroids:

        1/ Crop the image around the centroid,
        2/ Create Gaussian smoothed signal

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.signal_key = signal_key

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            ###########
            # Crop the image
            ###########
            d["current_label"] = list(d["centroids"][0].values())[0][-4]

            (
                x,
                y,
                z,
            ) = (
                list(d["centroids"][0].values())[0][-3],
                list(d["centroids"][0].values())[0][-2],
                list(d["centroids"][0].values())[0][-1],
            )
            current_size = d[key].shape[1:]
            original_size = d[key].meta["spatial_shape"]
            x = int(x * current_size[0] / original_size[0])
            y = int(y * current_size[1] / original_size[1])
            z = int(z * current_size[2] / original_size[2])

            # Cropping
            cropper = SpatialCrop(roi_center=[x, y, z], roi_size=(96, 96, 64))

            slices_cropped = [
                [cropper.slices[-3].start, cropper.slices[-3].stop],
                [cropper.slices[-2].start, cropper.slices[-2].stop],
                [cropper.slices[-1].start, cropper.slices[-1].stop],
            ]
            d["slices_cropped"] = slices_cropped
            d[key] = cropper(d[key])

            cropped_size = d[key].shape[1:]
            d["cropped_size"] = cropped_size

            #################################
            # Create signal based on centroid
            #################################
            signal = torch.zeros_like(d[key])
            signal[:, cropped_size[0] // 2, cropped_size[1] // 2, cropped_size[2] // 2] = 1.0

            sigma = 1.6 + (d["current_label"] - 1.0) * 0.1
            signal = GaussianSmooth(sigma)(signal)
            d[self.signal_key] = signal

        return d


class GetOriginalInformation(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Get information from original image

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]
        return d


class AddCentroidFromClicks(Transform, Randomizable):
    def __init__(self, label_names, key_label="label", key_clicks="foreground", key_centroids="centroids"):
        self.label_names = label_names
        self.key_label = key_label
        self.key_clicks = key_clicks
        self.key_centroids = key_centroids

    def __call__(self, data):
        d: Dict = dict(data)

        clicks = d.get(self.key_clicks, [])
        if clicks:
            label = d.get(self.key_label, "label")
            label_idx = self.label_names.get(label, 0)
            for click in clicks:
                d[self.key_centroids] = [{f"label_{label_idx}": [label_idx, click[-3], click[-2], click[-1]]}]

        logger.info(f"Using Centroid:  {label} => {d[self.key_centroids]}")
        return d


class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(self, keys: KeysCollection, label_names=None, allow_missing_keys: bool = False):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Dictionary containing new label numbers
            new_label_names = {}
            label = torch.zeros_like(d[key])
            # Making sure the range values and number of labels are the same
            idx = 1
            for key_label, val_label in self.label_names.items():
                if key_label != "background":
                    new_label_names[key_label] = idx
                    label[d[key] == val_label] = idx
                    idx += 1
                if key_label == "background":
                    new_label_names["background"] = 0

            d["label_names"] = new_label_names
            d[key].array = label
        return d


class CacheObjectd(MapTransform):
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            cache_key = f"{key}_cached"
            if d.get(cache_key) is None:
                d[cache_key] = copy.deepcopy(d[key])
        return d


def get_guidance_tensor_for_key_label(data, key_label, device) -> torch.Tensor:
    """Makes sure the guidance is in a tensor format."""
    tmp_gui = data.get(key_label, torch.tensor([], dtype=torch.int32, device=device))
    if isinstance(tmp_gui, list):
        tmp_gui = torch.tensor(tmp_gui, dtype=torch.int32, device=device)
    assert type(tmp_gui) is torch.Tensor or type(tmp_gui) is MetaTensor
    return tmp_gui


class AddGuidanceSignal(MapTransform):
    """
    Add Guidance signal for input image.

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        disks: This paraemters fill spheres with a radius of sigma centered around each click.
        device: device this transform shall run on.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 1,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.device = device

    def _get_corrective_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert (
            type(guidance) is torch.Tensor or type(guidance) is MetaTensor
        ), f"guidance is {type(guidance)}, value {guidance}"

        if guidance.size()[0]:
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                # Assuming the guidance has either shape (1, x, y , z) or (x, y, z)
                assert (
                    first_point_size == 4 or first_point_size == 3
                ), f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                assert first_point_size == 3, f"first_point_size is {first_point_size}, first_point is {guidance[0]}"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0  # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning(
                    "[WARNING] Bad signal values",
                    torch.min(signal[0]),
                    torch.max(signal[0]),
                )
            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros(
                    (1, image.shape[-3], image.shape[-2], image.shape[-1]),
                    device=self.device,
                )
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, Any]:
        for key in self.key_iterator(data):
            if key == "image":
                image = data[key]
                assert image.is_cuda
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                # e.g. {'spleen': '[[1, 202, 190, 192], [2, 224, 212, 192], [1, 242, 202, 192], [1, 256, 184, 192], [2.0, 258, 198, 118]]',
                # 'background': '[[257, 0, 98, 118], [1.0, 223, 303, 86]]'}

                for _, (label_key, _) in enumerate(data[LABELS_KEY].items()):
                    # label_guidance = data[label_key]
                    label_guidance = get_guidance_tensor_for_key_label(data, label_key, self.device)
                    logger.debug(f"Converting guidance for label {label_key}:{label_guidance} into a guidance signal..")

                    if label_guidance is not None and label_guidance.numel():
                        signal = self._get_corrective_signal(
                            image,
                            label_guidance.to(device=self.device),
                            key_label=label_key,
                        )
                        assert torch.sum(signal) > 0
                    else:
                        # TODO can speed this up here
                        signal = self._get_corrective_signal(
                            image,
                            torch.Tensor([]).to(device=self.device),
                            key_label=label_key,
                        )

                    assert signal.is_cuda
                    assert tmp_image.is_cuda
                    tmp_image = torch.cat([tmp_image, signal], dim=0)
                    if isinstance(data[key], MetaTensor):
                        data[key].array = tmp_image
                    else:
                        data[key] = tmp_image
                return data
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")


class AddEmptySignalChannels(MapTransform):
    def __init__(self, device, keys: KeysCollection = None):
        """
        Adds empty channels to the signal which will be filled with the guidance signal later.
        E.g. for two labels: 1x192x192x256 -> 3x192x192x256
        """
        super().__init__(keys)
        self.device = device

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, Any]:
        # Set up the initial batch data
        in_channels = 1 + len(data[LABELS_KEY])
        tmp_image = data[CommonKeys.IMAGE][0 : 0 + 1, ...]
        assert len(tmp_image.shape) == 4
        new_shape = list(tmp_image.shape)
        new_shape[0] = in_channels
        # Set the signal to 0 for all input images
        # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
        # e.g. (3,128,128,128) for two labels
        inputs = torch.zeros(new_shape, device=self.device)
        inputs[0] = data[CommonKeys.IMAGE][0]
        if isinstance(data[CommonKeys.IMAGE], MetaTensor):
            data[CommonKeys.IMAGE].array = inputs
        else:
            data[CommonKeys.IMAGE] = inputs

        return data
