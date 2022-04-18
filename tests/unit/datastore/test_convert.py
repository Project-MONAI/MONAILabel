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

import os
import tempfile
import unittest

import numpy as np
from monai.transforms import LoadImage

from monailabel.datastore.utils.convert import binary_to_image, dicom_to_nifti, nifti_to_dicom_seg


class TestConvert(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    local_dataset = os.path.join(base_dir, "data", "dataset", "local", "spleen")
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")

    def test_dicom_to_nifti(self):
        series_dir = os.path.join(self.dicom_dataset, "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087")
        result = dicom_to_nifti(series_dir)

        assert os.path.exists(result)
        assert result.endswith(".nii.gz")
        os.unlink(result)

    def test_binary_to_image(self):
        reference_image = os.path.join(self.local_dataset, "labels", "final", "spleen_3.nii.gz")
        label, _ = LoadImage()(reference_image)
        label = label.astype(np.uint16)
        label = label.flatten(order="F")

        label_bin = tempfile.NamedTemporaryFile(suffix=".bin").name
        label.tofile(label_bin)

        result = binary_to_image(reference_image, label_bin)
        os.unlink(label_bin)

        assert os.path.exists(result)
        assert result.endswith(".nii.gz")
        os.unlink(result)

    def test_nifti_to_dicom_seg(self):
        image = os.path.join(self.dicom_dataset, "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087")
        label = os.path.join(
            self.dicom_dataset,
            "labels",
            "final",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087.nii.gz",
        )
        result = nifti_to_dicom_seg(image, label, None, use_itk=False)

        assert os.path.exists(result)
        assert result.endswith(".dcm")
        os.unlink(result)

    def test_itk_image_to_dicom_seg(self):
        pass

    def test_itk_dicom_seg_to_image(self):
        pass


if __name__ == "__main__":
    unittest.main()
