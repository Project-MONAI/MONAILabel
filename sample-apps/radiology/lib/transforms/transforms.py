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

                # Plotting
                # from matplotlib.pyplot import imshow, show, close
                # imshow(d[key][0,:,:,int(d[key].shape[-1]/2)])
                # show()
                # close()
            elif key == "image":
                d[key] = cropper(d[key])
            else:
                print("This transform only applies to the label or image")

        # For debugging purposes
        # canvas_img = np.zeros(d["original_size"], dtype=np.float32)
        # canvas_label = np.zeros(d["original_size"], dtype=np.float32)
        #
        # canvas_img[
        #     cropper.slices[-3].start : cropper.slices[-3].stop,
        #     cropper.slices[-2].start : cropper.slices[-2].stop,
        #     cropper.slices[-1].start : cropper.slices[-1].stop,
        # ] = d["image"]
        #
        # canvas_label[
        #     cropper.slices[-3].start : cropper.slices[-3].stop,
        #     cropper.slices[-2].start : cropper.slices[-2].stop,
        #     cropper.slices[-1].start : cropper.slices[-1].stop,
        # ] = d["label"]
        #
        # d["image"] = canvas_img
        # d["label"] = canvas_label
        #
        # SaveImaged(keys="image", output_postfix="", output_dir="/home/andres/Downloads", separate_folder=False)(d)
        # SaveImaged(keys="label", output_postfix="seg", output_dir="/home/andres/Downloads", separate_folder=False)(d)

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
            canvas_img = np.zeros(d["original_size"], dtype=np.float32)
            if key == "pred":
                # How to get the ROI to reconstruct final image
                # Iterate over all the vertebras
                print(d[key].shape)
            else:
                print("This transform only applies to the pred")
        return d


# For the second stage - Vertebra localization


class VertHeatMap(MapTransform):
    def __init__(self, keys, sigma=3.0, label_names=None):
        super().__init__(keys)
        self.sigma = sigma
        self.label_names = label_names

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
                out.fill_(0)

            # loop over all segmentation classes
            for seg_class in torch.unique(i):
                # skip background
                if seg_class == 0:
                    continue
                # get CoM for given segmentation class
                centre = [np.average(indices.cpu()).astype(int) for indices in torch.where(i[0] == seg_class)]
                centre.insert(0, seg_class.item())
                out[tuple(centre)] = 1

            # TO DO: Keep the centroids in the data dictionary!

            # Gaussian smooth
            out = GaussianSmooth(self.sigma, "scalespace")(out.cuda()).cpu()

            # Normalize to [0,1]
            out = ScaleIntensity()(out)

            # Fill in background
            out[0] = 1 - out[1:].sum(0)
            out = torch.clamp(out, min=0)

            data[k] = out

            # SaveImaged(keys="label", output_postfix="", output_dir="/home/andres/Downloads", separate_folder=False)(data)

        return data
