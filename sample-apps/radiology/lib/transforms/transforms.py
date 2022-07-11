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

# from monai.transforms import SaveImaged, SpatialCrop
from monai.transforms.transform import MapTransform

logger = logging.getLogger(__name__)


class HeatMapROId(MapTransform):
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
            if key == "label":
                # Convert to single label
                d[key][d[key] > 0] = 1
            else:
                print("This transform only applies to the label")
        return d


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
            if key == "label":
                # Convert to single label
                d[key][d[key] > 0] = 1
            else:
                print("This transform only applies to the label")
        return d


class GetCentroidAndCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size=None,
        allow_missing_keys: bool = False,
    ):
        """
        Compute centroid

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        if roi_size is None:
            roi_size = [256, 256, 256]
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
        current_label = 0
        padd = 100
        for key in self.key_iterator(d):
            if key == "label":

                centroids = self._getCentroids(d[key])
                d["centroids"] = centroids

                d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]

                # TO DO: SELECT A DIFFERENT FOR EACH ITERATION
                firstLabel = d["centroids"][current_label]

                point = []
                point.append(firstLabel["X"])
                point.append(firstLabel["Y"])
                point.append(firstLabel["Z"])

                # Cropping
                # cropper = SpatialCrop(roi_center=point, roi_size=self.roi_size)
                # d[key] = cropper(d[key])
                # d[key] = d[key][0, 100:-1, 100:-1, 90:-1]

                d[key] = d[key][
                    0,
                    point[-3] - padd : point[-3] + padd,
                    point[-2] - padd : point[-2] + padd,
                    point[-1] - int(padd / 4) : point[-1] + int(padd / 4),
                ][None]

                # Make the cropping binary
                d[key][d[key] != firstLabel["label"]] = 0

                # Plotting
                # from matplotlib.pyplot import imshow, show, close
                # imshow(d[key][0,:,:,int(d[key].shape[-1]/2)])
                # show()
                # close()

            elif key == "image":
                # d[key] = cropper(d[key])
                # d[key] = d[key][0, 100:-1, 100:-1, 90:-1]
                d[key] = d[key][
                    0,
                    point[-3] - padd : point[-3] + padd,
                    point[-2] - padd : point[-2] + padd,
                    point[-1] - int(padd / 4) : point[-1] + int(padd / 4),
                ][None]

                # Plotting
                # from matplotlib.pyplot import imshow, show, close
                # imshow(d[key][:,:,60])
                # show()
                # close()
            else:
                print("This transform only applies to the label or image")

        # For debugging
        # canvas_img = np.zeros(d["original_size"], dtype=np.float32)
        # canvas_label = np.zeros(d["original_size"], dtype=np.float32)
        #
        # canvas_img[
        #     point[-3] - padd : point[-3] + padd,
        #     point[-2] - padd : point[-2] + padd,
        #     point[-1] - int(padd / 4) : point[-1] + int(padd / 4),
        # ] = d["image"]
        #
        # canvas_label[
        #     point[-3] - padd : point[-3] + padd,
        #     point[-2] - padd : point[-2] + padd,
        #     point[-1] - int(padd / 4) : point[-1] + int(padd / 4),
        # ] = d["label"]
        #
        # d["image"] = canvas_img
        # d["label"] = canvas_label

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
            if key == "label":

                point = d["centroid"]
                logger.info("Processing label: " + d["label_meta_dict"]["filename_or_obj"])
                signal = np.zeros((1, d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]), dtype=np.float32)
                sshape = signal.shape
                # Making sure points fall inside the image dimension
                p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                signal[:, p1, p2, p3] = 1.0

                # Apply a Gaussian filter to the signal
                signal_tensor = torch.tensor(signal[0])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[0] = signal_tensor.detach().cpu().numpy()
                signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))
                d["signal"] = signal

                # # Plotting
                # from matplotlib.pyplot import imshow, show, close
                # imshow(d['signal'][0,:,:,60])
                # show()
                # close()

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
                # How to pass information from the pre-transform - the ROI specifically
                # canvas_img[
                # point[-3] - padd: point[-3] + padd,
                # point[-2] - padd: point[-2] + padd,
                # point[-1] - int(padd / 4): point[-1] + int(padd / 4),
                # ] = d["pred"]
                d["pred"] = canvas_img
            else:
                print("This transform only applies to the pred")
        return d
