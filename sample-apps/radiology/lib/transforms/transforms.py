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
from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms import CropForeground, GaussianSmooth, Randomizable, Resize, ScaleIntensity, SpatialCrop
from monai.transforms.transform import MapTransform, Transform

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

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[key][d[key] > 0] = 1
        return d


class SelectVertebraAndCroppingd(Randomizable, MapTransform):
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

            slices_cropped = [
                [start[-3], stop[-3]],
                [start[-2], stop[-2]],
                [start[-1], stop[-1]],
            ]

            d["slices_cropped"] = slices_cropped

            # Cropping label
            d["label"] = tmp_label[:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

            # Cropping image
            d["image"] = d["image"][:, start[-3] : stop[-3], start[-2] : stop[-2], start[-1] : stop[-1]]

        return d


class GetCentroidsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        centroids_key: str = "centroids",
        allow_missing_keys: bool = False,
    ):
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
    def __init__(
        self,
        keys: KeysCollection,
        signal_key: str = "signal",
        allow_missing_keys: bool = False,
    ):
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

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            tmp_image = np.concatenate([d["image"], d[key]], axis=0)
            d["image"] = tmp_image
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
    def __init__(
        self,
        keys: KeysCollection,
        signal_key,
        allow_missing_keys: bool = False,
    ):
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

            x, y, z, = (
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
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
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
            for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                if key_label != "background":
                    new_label_names[key_label] = idx
                    label[d[key] == val_label] = idx
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
