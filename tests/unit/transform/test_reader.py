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
import unittest
from pathlib import Path

import numpy as np
from monai.transforms import LoadImage

# Check if required dependencies are available
try:
    from nvidia import nvimgcodec

    HAS_NVIMGCODEC = True
except ImportError:
    HAS_NVIMGCODEC = False
    nvimgcodec = None

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    pydicom = None

# Import the reader
try:
    from monailabel.transform.reader import NvDicomReader

    HAS_NVDICOMREADER = True
except ImportError:
    HAS_NVDICOMREADER = False
    NvDicomReader = None


@unittest.skipIf(not HAS_NVDICOMREADER, "NvDicomReader not available")
@unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
class TestNvDicomReader(unittest.TestCase):
    """Test suite for NvDicomReader with HTJ2K encoded DICOM files."""

    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")

    # Test series for HTJ2K decoding
    test_series_uid = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721"

    def setUp(self):
        """Set up test fixtures."""
        # Paths to test data
        self.original_series_dir = os.path.join(self.dicom_dataset, self.test_series_uid)
        self.htj2k_series_dir = os.path.join(
            self.base_dir, "data", "dataset", "dicomweb_htj2k", "e7567e0a064f0c334226a0658de23afd", self.test_series_uid
        )
        self.reference_nifti = os.path.join(self.dicom_dataset, f"{self.test_series_uid}.nii.gz")

    def _check_test_data(self, directory, desc="DICOM"):
        """Check if test data exists."""
        if not os.path.exists(directory):
            return False
        dcm_files = list(Path(directory).glob("*.dcm"))
        if len(dcm_files) == 0:
            return False
        return True

    def _get_reference_image(self):
        """Load reference NIfTI image."""
        if not os.path.exists(self.reference_nifti):
            self.fail(f"Reference NIfTI file not found: {self.reference_nifti}")

        loader = LoadImage(image_only=False)
        img_array, meta = loader(self.reference_nifti)
        # Reference NIfTI is in (W, H, D) order
        return np.array(img_array), meta

    def test_nvdicomreader_original_series(self):
        """Test NvDicomReader with original (non-HTJ2K) DICOM series."""
        # Check test data exists
        if not self._check_test_data(self.original_series_dir, "original DICOM"):
            self.skipTest(f"Original DICOM test data not found at {self.original_series_dir}")

        # Load with NvDicomReader (use reverse_indexing=True to match NIfTI W,H,D layout)
        reader = NvDicomReader(reverse_indexing=True)
        img_obj = reader.read(self.original_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify shape (should be W, H, D with reverse_indexing=True)
        self.assertEqual(volume.shape, (512, 512, 77), f"Expected shape (512, 512, 77), got {volume.shape}")

        # Load reference NIfTI for comparison
        reference, ref_meta = self._get_reference_image()

        # Compare with reference (allowing for small numerical differences)
        np.testing.assert_allclose(
            volume, reference, rtol=1e-5, atol=1e-3, err_msg="NvDicomReader output differs from reference NIfTI"
        )

        print(f"✓ NvDicomReader original DICOM series test passed")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available for HTJ2K decoding")
    def test_nvdicomreader_htj2k_series(self):
        """Test NvDicomReader with HTJ2K-encoded DICOM series."""
        # Check HTJ2K test data exists
        if not self._check_test_data(self.htj2k_series_dir, "HTJ2K DICOM"):
            # Try to create HTJ2K data if nvimgcodec is available
            print("\nHTJ2K test data not found. Attempting to create...")
            import sys

            sys.path.insert(0, os.path.join(self.base_dir))
            try:
                from prepare_htj2k_test_data import create_htj2k_data

                create_htj2k_data(os.path.join(self.base_dir, "data"))
            except Exception as e:
                self.skipTest(f"Could not create HTJ2K test data: {e}")

            # Re-check after creation attempt
            if not self._check_test_data(self.htj2k_series_dir, "HTJ2K DICOM"):
                self.skipTest(f"HTJ2K DICOM files not found at {self.htj2k_series_dir}")

        # Verify these are actually HTJ2K encoded
        htj2k_files = list(Path(self.htj2k_series_dir).glob("*.dcm"))
        first_dcm = pydicom.dcmread(str(htj2k_files[0]))
        transfer_syntax = first_dcm.file_meta.TransferSyntaxUID
        htj2k_syntaxes = [
            "1.2.840.10008.1.2.4.201",  # HTJ2K Lossless
            "1.2.840.10008.1.2.4.202",  # HTJ2K with RPCL
            "1.2.840.10008.1.2.4.203",  # HTJ2K Lossy
        ]
        if str(transfer_syntax) not in htj2k_syntaxes:
            self.skipTest(f"DICOM files are not HTJ2K encoded (Transfer Syntax: {transfer_syntax})")

        # Load with NvDicomReader (use reverse_indexing=True to match NIfTI W,H,D layout)
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False, reverse_indexing=True)
        img_obj = reader.read(self.htj2k_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify shape (should be W, H, D with reverse_indexing=True)
        self.assertEqual(volume.shape, (512, 512, 77), f"Expected shape (512, 512, 77), got {volume.shape}")

        # Load reference NIfTI for comparison
        reference, ref_meta = self._get_reference_image()

        # Convert to numpy if cupy array (batch decode may return GPU arrays)
        if hasattr(volume, "__cuda_array_interface__"):
            import cupy as cp

            volume = cp.asnumpy(volume)

        # Compare with reference (HTJ2K is lossless, so should be identical)
        np.testing.assert_allclose(
            volume, reference, rtol=1e-5, atol=1e-3, err_msg="HTJ2K decoded volume differs from reference NIfTI"
        )

        print(f"✓ NvDicomReader HTJ2K DICOM series test passed")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available for HTJ2K decoding")
    def test_htj2k_vs_original_consistency(self):
        """Test that HTJ2K decoding produces the same result as original DICOM."""
        # Check both datasets exist
        if not self._check_test_data(self.original_series_dir, "original DICOM"):
            self.skipTest(f"Original DICOM test data not found at {self.original_series_dir}")

        if not self._check_test_data(self.htj2k_series_dir, "HTJ2K DICOM"):
            # Try to create HTJ2K data
            print("\nHTJ2K test data not found. Attempting to create...")
            import sys

            sys.path.insert(0, os.path.join(self.base_dir))
            try:
                from prepare_htj2k_test_data import create_htj2k_data

                create_htj2k_data(os.path.join(self.base_dir, "data"))
            except Exception as e:
                self.skipTest(f"Could not create HTJ2K test data: {e}")

            # Re-check after creation attempt
            if not self._check_test_data(self.htj2k_series_dir, "HTJ2K DICOM"):
                self.skipTest(f"HTJ2K DICOM files not found at {self.htj2k_series_dir}")

        # Load original series (use reverse_indexing=True for W,H,D layout)
        reader_original = NvDicomReader(use_nvimgcodec=False, reverse_indexing=True)  # Force pydicom for original
        img_obj_orig = reader_original.read(self.original_series_dir)
        volume_orig, metadata_orig = reader_original.get_data(img_obj_orig)

        # Load HTJ2K series with nvImageCodec (use reverse_indexing=True for W,H,D layout)
        reader_htj2k = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False, reverse_indexing=True)
        img_obj_htj2k = reader_htj2k.read(self.htj2k_series_dir)
        volume_htj2k, metadata_htj2k = reader_htj2k.get_data(img_obj_htj2k)

        # Convert to numpy if cupy arrays
        if hasattr(volume_orig, "__cuda_array_interface__"):
            import cupy as cp

            volume_orig = cp.asnumpy(volume_orig)
        if hasattr(volume_htj2k, "__cuda_array_interface__"):
            import cupy as cp

            volume_htj2k = cp.asnumpy(volume_htj2k)

        # Verify shapes match
        self.assertEqual(volume_orig.shape, volume_htj2k.shape, "Original and HTJ2K volumes should have the same shape")

        # Compare volumes (HTJ2K lossless should be identical)
        np.testing.assert_allclose(
            volume_orig, volume_htj2k, rtol=1e-5, atol=1e-3, err_msg="HTJ2K decoded volume differs from original DICOM"
        )

        # Verify metadata consistency
        self.assertEqual(
            metadata_orig["spacing"].tolist(), metadata_htj2k["spacing"].tolist(), "Spacing should be identical"
        )

        np.testing.assert_allclose(
            metadata_orig["affine"], metadata_htj2k["affine"], rtol=1e-6, err_msg="Affine matrices should be identical"
        )

        print(f"✓ HTJ2K vs original consistency test passed")

    def test_nvdicomreader_metadata(self):
        """Test that NvDicomReader extracts proper metadata."""
        if not self._check_test_data(self.original_series_dir):
            self.skipTest(f"Original DICOM test data not found at {self.original_series_dir}")

        reader = NvDicomReader(reverse_indexing=True)
        img_obj = reader.read(self.original_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Check essential metadata fields
        self.assertIn("affine", metadata, "Metadata should contain affine matrix")
        self.assertIn("spacing", metadata, "Metadata should contain spacing")
        self.assertIn("spatial_shape", metadata, "Metadata should contain spatial_shape")

        # Verify affine is 4x4
        self.assertEqual(metadata["affine"].shape, (4, 4), "Affine should be 4x4")

        # Verify spacing has 3 elements
        self.assertEqual(len(metadata["spacing"]), 3, "Spacing should have 3 elements")

        # Verify spatial shape matches volume shape
        np.testing.assert_array_equal(
            metadata["spatial_shape"], volume.shape, err_msg="Spatial shape in metadata should match volume shape"
        )

        print(f"✓ NvDicomReader metadata test passed")

    def test_nvdicomreader_reverse_indexing(self):
        """Test NvDicomReader with reverse_indexing=True (ITK-style layout)."""
        if not self._check_test_data(self.original_series_dir):
            self.skipTest(f"Original DICOM test data not found at {self.original_series_dir}")

        # Default: reverse_indexing=False -> (depth, height, width)
        reader_default = NvDicomReader(reverse_indexing=False)
        img_obj_default = reader_default.read(self.original_series_dir)
        volume_default, _ = reader_default.get_data(img_obj_default)

        # ITK-style: reverse_indexing=True -> (width, height, depth)
        reader_itk = NvDicomReader(reverse_indexing=True)
        img_obj_itk = reader_itk.read(self.original_series_dir)
        volume_itk, _ = reader_itk.get_data(img_obj_itk)

        # Verify shapes are transposed correctly
        self.assertEqual(volume_default.shape, (77, 512, 512))
        self.assertEqual(volume_itk.shape, (512, 512, 77))

        # Verify data is the same (just transposed)
        np.testing.assert_allclose(
            volume_default.transpose(2, 1, 0),
            volume_itk,
            rtol=1e-6,
            err_msg="Reverse indexing should produce transposed volume",
        )

        print(f"✓ NvDicomReader reverse_indexing test passed")


@unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available")
@unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
class TestNvDicomReaderHTJ2KPerformance(unittest.TestCase):
    """Performance tests for HTJ2K decoding with NvDicomReader."""

    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")
    test_series_uid = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721"

    def setUp(self):
        """Set up test fixtures."""
        self.htj2k_series_dir = os.path.join(
            self.base_dir, "data", "dataset", "dicomweb_htj2k", "e7567e0a064f0c334226a0658de23afd", self.test_series_uid
        )

    def test_batch_decode_optimization(self):
        """Test that batch decode is used for HTJ2K series."""
        # Skip if HTJ2K data not available
        if not os.path.exists(self.htj2k_series_dir):
            self.skipTest(f"HTJ2K test data not found at {self.htj2k_series_dir}")

        htj2k_files = list(Path(self.htj2k_series_dir).glob("*.dcm"))
        if len(htj2k_files) == 0:
            self.skipTest(f"No HTJ2K DICOM files found in {self.htj2k_series_dir}")

        # Verify HTJ2K encoding
        first_dcm = pydicom.dcmread(str(htj2k_files[0]))
        transfer_syntax = str(first_dcm.file_meta.TransferSyntaxUID)
        htj2k_syntaxes = ["1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202", "1.2.840.10008.1.2.4.203"]
        if transfer_syntax not in htj2k_syntaxes:
            self.skipTest(f"DICOM files are not HTJ2K encoded")

        # Load with batch decode enabled
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.htj2k_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify successful decode
        self.assertIsNotNone(volume, "Volume should be decoded successfully")
        self.assertEqual(volume.shape[0], len(htj2k_files), f"Volume should have {len(htj2k_files)} slices")

        print(f"✓ Batch decode optimization test passed ({len(htj2k_files)} slices)")


if __name__ == "__main__":
    unittest.main()
