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

import logging
from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.networks.layers import GaussianFilter
from monai.transforms import GaussianSmooth, ScaleIntensity, SpatialCrop
from monai.transforms.transform import MapTransform

logger = logging.getLogger(__name__)


class BinaryMaskd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        """
        Convert to single label - This should actually create the heat map for the first stage

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[key][d[key] > 0] = 1
        return d


class GetCentroidAndCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size=None,
        allow_missing_keys: bool = False,
    ):
        """
        Compute centroids and crop image and label

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        if roi_size is None:
            roi_size = [128, 128, 128]
        self.roi_size = roi_size

    def _getCentroids(self, label):
        centroids = []
        # loop over all segments
        for seg_class in np.unique(label):
            c = {}
            # skip background
            if seg_class == 0:
                continue
            # get centre of mass (CoM)
            centre = [np.average(indices).astype(int) for indices in np.where(label == seg_class)]
            c["label"] = int(seg_class)
            c["X"] = centre[-3]
            c["Y"] = centre[-2]
            c["Z"] = centre[-1]
            centroids.append(c)
        return centroids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Logic:

            # 1/ get centroids
            # 2/ randomly select vertebra
            # 3/ spatial crop based on the centroid
            # 4/ binarise cropped volume

            # Alternative logic:

            # 1/ get centroids
            # 2/ randomly select vertebra
            # 3/ binarise volume
            # 4/ apply cropForeground transform

            if key == "label":

                # Get centroids
                d["centroids"] = self._getCentroids(d[key])

                d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]

                # TO DO: WHAT'S THE BEST WAY TO SELECT A DIFFERENT SEGMENT EACH ITERATION
                # PERHAPS DOING BATCHES AS IN TRANSFORM RandCropByPosNegLabeld
                current_label = np.random.randint(0, len(d["centroids"]))
                first_label = d["centroids"][current_label]
                d["current_label"] = current_label

                logger.info(f"Processing vertebra: {first_label['label']}")

                centroid = [first_label["X"], first_label["Y"], first_label["Z"]]

                # Cropping
                cropper = SpatialCrop(roi_center=centroid, roi_size=self.roi_size)

                slices_cropped = [
                    [cropper.slices[-3].start, cropper.slices[-3].stop],
                    [cropper.slices[-2].start, cropper.slices[-2].stop],
                    [cropper.slices[-1].start, cropper.slices[-1].stop],
                ]

                d["slices_cropped"] = slices_cropped

                d[key] = cropper(d[key])

                # Make binary the cropped label
                d[key][d[key] != first_label["label"]] = 0
                d[key][d[key] > 0] = 1

            elif key == "image":
                d[key] = cropper(d[key])
            else:
                print("This transform only applies to the label or image")

        return d


class GaussianSmoothedCentroidd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 5,
        allow_missing_keys: bool = False,
    ):
        """
        Apply Gaussian to Centroid

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                logger.info("Processing label: " + d["label_meta_dict"]["filename_or_obj"])

                c_label = d["centroids"][d["current_label"]]
                signal = np.zeros(d["original_size"], dtype=np.float32)
                signal[c_label["X"], c_label["Y"], c_label["Z"]] = 1.0
                signal = signal[
                    d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
                    d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
                    d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
                ]
                signal = signal[None]

                # Apply a Gaussian filter to the signal
                signal_tensor = torch.tensor(signal[0])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[0] = signal_tensor.detach().cpu().numpy()
                signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))
                d["signal"] = signal

            else:
                print("This transform only applies to the signal")
        return d


class AddROI(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        """
        Add Gaussian smoothed centroid (signal) to cropped volume

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "signal":
                tmp_image = np.concatenate([d["image"], d[key]], axis=0)
                d["image"] = tmp_image
            else:
                print("This transform only applies to the signal")
        return d


class PlaceCroppedAread(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        """
        Place the ROI predicted in the full image

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            final_pred = np.zeros(d["original_size"], dtype=np.float32)
            if key == "pred":
                final_pred[
                    d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
                    d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
                    d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
                ] = d["pred"]
                d["pred"] = final_pred * int(d["current_label"])
                # How to get the ROI to reconstruct final image? - Iterate over all the vertebras
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
    def __init__(
        self,
        keys: KeysCollection,
        result: str = "result",
        allow_missing_keys: bool = False,
    ):
        """
        Postprocess Vertebra localization

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.result = result

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        centroids = []
        for key in self.key_iterator(d):

            # Getting centroids
            for l in range(d[key].shape[0] - 1):
                centroid = {}
                if d[key][l + 1, ...].max() < 30.0:
                    continue
                X, Y, Z = np.where(d[key][l + 1, ...] == d[key][l + 1, ...].max())
                X, Y, Z = X[0], Y[0], Z[0]
                centroid[f"label_{l+1}"] = [X, Y, Z]
                centroids.append(centroid)

            print(centroids)
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result]["centroids"] = centroids
        return d


class AddROIThirdStage(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 2,
        allow_missing_keys: bool = False,
    ):
        """
        Based on the centroids:

        1/ Crop the image around the centroid,
        2/ Create Gaussian smoothed signal
        3/ Concatenate signal to the cropped volume

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":

                ###########
                # Crop the image
                ###########

                d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]
                current_label = np.random.randint(0, len(d["centroids"]))
                d["current_label"] = list(d["centroids"][current_label].values())[0][-4]
                X, Y, Z, = (
                    list(d["centroids"][current_label].values())[0][-3],
                    list(d["centroids"][current_label].values())[0][-2],
                    list(d["centroids"][current_label].values())[0][-1],
                )
                centroid = [X, Y, Z]
                # Cropping
                cropper = SpatialCrop(roi_center=centroid, roi_size=(128, 128, 96))
                slices_cropped = [
                    [cropper.slices[-3].start, cropper.slices[-3].stop],
                    [cropper.slices[-2].start, cropper.slices[-2].stop],
                    [cropper.slices[-1].start, cropper.slices[-1].stop],
                ]
                d["slices_cropped"] = slices_cropped
                d[key] = cropper(d[key])
                # Smooth the image as it was done during training
                d[key] = GaussianSmooth(sigma=0.75)(d[key])

                #################################
                # Create signal based on centroid
                #################################

                signal = np.zeros(d["original_size"], dtype=np.float32)
                signal[X, Y, Z] = 1.0
                signal = signal[
                    d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
                    d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
                    d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
                ]
                signal = signal[None]

                # Apply a Gaussian filter to the signal
                signal_tensor = torch.tensor(signal[0])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[0] = signal_tensor.detach().cpu().numpy()
                signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))

                ##################################
                # Concatenate signal with centroid
                ##################################
                tmp_image = np.concatenate([d[key], signal], axis=0)
                d[key] = tmp_image

        return d
