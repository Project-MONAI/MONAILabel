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
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms import CropForeground, GaussianSmooth, ScaleIntensity, SpatialCrop
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


class HeuristicCroppingd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        """
        Crop image and label

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):

            # Alternative Logic:

            # 1/ randomly select vertebra
            # 2/ spatial crop based on the centroid
            # 3/ binarise cropped volume

            # logic:

            # 1/ randomly select vertebra
            # 2/ binarise volume
            # 3/ apply cropForeground transform

            if key == "label":
                d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]

                # TO DO: WHAT'S THE BEST WAY TO SELECT A DIFFERENT SEGMENT EACH ITERATION
                # PERHAPS DOING BATCHES AS IN TRANSFORM RandCropByPosNegLabeld ??

                current_idx = np.random.randint(0, len(d["centroids"]))

                d["current_idx"] = current_idx

                d["current_label"] = list(d["centroids"][current_idx].values())[0][0]

                logger.info(f'Processing vertebra: {d["current_label"]}')

                # Make binary the label
                d["label"][d["label"] != d["current_label"]] = 0
                d["label"][d["label"] > 0] = 1

                ##########
                # Cropping
                ##########
                def condition(x):
                    # threshold at 1
                    return x > 0

                cropper = CropForeground(select_fn=condition, margin=8, k_divisible=32)

                start, stop = cropper.compute_bounding_box(d["label"])

                slices_cropped = [
                    [start[-3], stop[-3]],
                    [start[-2], stop[-2]],
                    [start[-1], stop[-1]],
                ]

                d["slices_cropped"] = slices_cropped

                # Cropping label
                d["label"] = d["label"][:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

                # Cropping image
                d["image"] = d["image"][:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

        return d


class GetCentroidsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        centroids: str = "centroids",
        allow_missing_keys: bool = False,
    ):
        """
        Get centroids

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.centroids = centroids

    def _getCentroids(self, label):
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Get centroids
            d[self.centroids] = self._getCentroids(d[key])
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

                signal = np.zeros(
                    (1, d["original_size"][-3], d["original_size"][-2], d["original_size"][-1]), dtype=np.float32
                )

                X, Y, Z = (
                    list(d["centroids"][d["current_idx"]].values())[0][-3],
                    list(d["centroids"][d["current_idx"]].values())[0][-2],
                    list(d["centroids"][d["current_idx"]].values())[0][-1],
                )
                signal[:, X, Y, Z] = 1.0

                signal = GaussianSmooth(self.sigma)(signal)

                signal = signal[
                    :,
                    d["slices_cropped"][-3][0] : d["slices_cropped"][-3][1],
                    d["slices_cropped"][-2][0] : d["slices_cropped"][-2][1],
                    d["slices_cropped"][-1][0] : d["slices_cropped"][-1][1],
                ]

                d["signal"] = signal * d["label"]

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
            final_pred = np.zeros(
                (1, d["original_size"][-3], d["original_size"][-2], d["original_size"][-1]), dtype=np.float32
            )
            if key == "pred":
                final_pred[
                    :,
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
                centroid[f"label_{l + 1}"] = [X, Y, Z]
                centroids.append(centroid)

            print(centroids)
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result]["centroids"] = centroids
        return d


class VertebraLocalizationSegmentation(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        result: str = "result",
        allow_missing_keys: bool = False,
    ):
        """
        Postprocess Vertebra localization using segmentation task

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.result = result

    def _getCentroids(self, label):
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
                # most_indices = np.percentile(indices, 60).astype(int).tolist()
                # centre.append(most_indices)
                avg_indices = np.average(indices).astype(int)
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

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        centroids = []
        for key in self.key_iterator(d):
            # Getting centroids
            centroids = self._getCentroids(d[key])
            if d.get(self.result) is None:
                d[self.result] = dict()
            d[self.result]["centroids"] = centroids
        return d


class AddROIThirdStage(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sigma: int = 5,
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
                # d["current_label"] = list(d["centroids"]["centroids"][current_label].values())[0][-4]
                # X, Y, Z, = (
                #     list(d["centroids"]["centroids"][current_label].values())[0][-3],
                #     list(d["centroids"]["centroids"][current_label].values())[0][-2],
                #     list(d["centroids"]["centroids"][current_label].values())[0][-1],
                # )
                d["current_label"] = list(d["centroids"][current_label].values())[0][-4]
                X, Y, Z, = (
                    list(d["centroids"][current_label].values())[0][-3],
                    list(d["centroids"][current_label].values())[0][-2],
                    list(d["centroids"][current_label].values())[0][-1],
                )
                centroid = [X, Y, Z]
                # Cropping
                cropper = SpatialCrop(roi_center=centroid, roi_size=(128, 128, 128))

                slices_cropped = [
                    [cropper.slices[-3].start, cropper.slices[-3].stop],
                    [cropper.slices[-2].start, cropper.slices[-2].stop],
                    [cropper.slices[-1].start, cropper.slices[-1].stop],
                ]
                d["slices_cropped"] = slices_cropped
                d[key] = cropper(d[key])

                # Smooth the image as it was done during training
                d[key] = GaussianSmooth(sigma=0.75)(d[key])

                # Scale image intensity as it was done during training
                d[key] = ScaleIntensity(minv=-1.0, maxv=1.0)(d[key])

                #################################
                # Create signal based on centroid
                #################################
                signal = np.zeros(
                    (1, d["original_size"][-3], d["original_size"][-2], d["original_size"][-1]), dtype=np.float32
                )
                # X, Y, Z = (
                #     list(d["centroids"]["centroids"][current_label].values())[0][-3],
                #     list(d["centroids"]["centroids"][current_label].values())[0][-2],
                #     list(d["centroids"]["centroids"][current_label].values())[0][-1],
                # )
                X, Y, Z = (
                    list(d["centroids"][current_label].values())[0][-3],
                    list(d["centroids"][current_label].values())[0][-2],
                    list(d["centroids"][current_label].values())[0][-1],
                )
                signal[:, X, Y, Z] = 1.0
                signal = GaussianSmooth(self.sigma)(signal)
                signal = signal[
                    :,
                    cropper.slices[-3].start : cropper.slices[-3].stop,
                    cropper.slices[-2].start : cropper.slices[-2].stop,
                    cropper.slices[-1].start : cropper.slices[-1].stop,
                ]

                ##################################
                # Concatenate signal with centroid
                ##################################
                tmp_image = np.concatenate([d[key], signal], axis=0)
                d[key] = tmp_image

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
            for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                if key_label != "background":
                    new_label_names[key_label] = idx
                    label[d[key] == val_label] = idx
                if key_label == "background":
                    new_label_names["background"] = 0

            d["label_names"] = new_label_names
            d[key].array = label
        return d
