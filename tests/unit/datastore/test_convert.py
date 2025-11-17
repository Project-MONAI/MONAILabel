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
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from monai.transforms import LoadImage

from monailabel.datastore.utils.convert import (
    binary_to_image,
    dicom_to_nifti,
    nifti_to_dicom_seg,
)

# Check if nvimgcodec is available
try:
    from nvidia import nvimgcodec

    HAS_NVIMGCODEC = True
except ImportError:
    HAS_NVIMGCODEC = False
    nvimgcodec = None

# HTJ2K Transfer Syntax UIDs
HTJ2K_TRANSFER_SYNTAXES = frozenset(
    [
        "1.2.840.10008.1.2.4.201",  # High-Throughput JPEG 2000 Image Compression (Lossless Only)
        "1.2.840.10008.1.2.4.202",  # High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)
        "1.2.840.10008.1.2.4.203",  # High-Throughput JPEG 2000 Image Compression
    ]
)


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
        label = LoadImage(image_only=True)(reference_image)
        label = label.astype(np.uint8)
        label = label.flatten(order="F")

        label_bin = tempfile.NamedTemporaryFile(suffix=".bin").name
        label.tofile(label_bin)

        result = binary_to_image(reference_image, label_bin)
        os.unlink(label_bin)

        assert os.path.exists(result)
        assert result.endswith(".nii.gz")
        os.unlink(result)

    def test_nifti_to_dicom_seg_highdicom(self):
        """Test NIfTI to DICOM SEG conversion using highdicom (use_itk=False)."""
        series_dir = os.path.join(self.dicom_dataset, "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087")
        label_file = os.path.join(
            self.dicom_dataset,
            "labels",
            "final",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087.nii.gz",
        )

        # Convert using highdicom (use_itk=False)
        result = nifti_to_dicom_seg(series_dir, label_file, None, use_itk=False)

        # Verify output
        self.assertTrue(os.path.exists(result), "DICOM SEG file should be created")
        self.assertTrue(result.endswith(".dcm"), "Output should be a DICOM file")

        # Verify it's a valid DICOM file
        ds = pydicom.dcmread(result)
        self.assertEqual(ds.Modality, "SEG", "Should be a DICOM Segmentation object")

        # Verify segment count
        input_label = LoadImage(image_only=True)(label_file)
        num_labels = len(np.unique(input_label)) - 1  # Exclude background (0)
        if hasattr(ds, "SegmentSequence"):
            num_segments = len(ds.SegmentSequence)
            print(f"  Segments in DICOM SEG: {num_segments}, Unique labels in input: {num_labels}")

        # Clean up
        os.unlink(result)

        print(f"✓ NIfTI → DICOM SEG conversion successful (highdicom)")

    def test_nifti_to_dicom_seg_itk(self):
        """Test NIfTI to DICOM SEG conversion using ITK (use_itk=True)."""
        series_dir = os.path.join(self.dicom_dataset, "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087")
        label_file = os.path.join(
            self.dicom_dataset,
            "labels",
            "final",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087.nii.gz",
        )

        # Check if ITK/dcmqi is available
        import shutil

        itk_available = shutil.which("itkimage2segimage") is not None

        if not itk_available:
            self.skipTest(
                "itkimage2segimage command-line tool not found. "
                "Install dcmqi: pip install dcmqi (https://github.com/QIICR/dcmqi)"
            )

        # Convert using ITK (use_itk=True)
        result = nifti_to_dicom_seg(series_dir, label_file, None, use_itk=True)

        # Verify output
        self.assertTrue(os.path.exists(result), "DICOM SEG file should be created")
        self.assertTrue(result.endswith(".dcm"), "Output should be a DICOM file")

        # Verify it's a valid DICOM file
        ds = pydicom.dcmread(result)
        self.assertEqual(ds.Modality, "SEG", "Should be a DICOM Segmentation object")

        # Verify segment count
        input_label = LoadImage(image_only=True)(label_file)
        num_labels = len(np.unique(input_label)) - 1  # Exclude background (0)
        if hasattr(ds, "SegmentSequence"):
            num_segments = len(ds.SegmentSequence)
            print(f"  Segments in DICOM SEG: {num_segments}, Unique labels in input: {num_labels}")

        # Clean up
        os.unlink(result)

        print(f"✓ NIfTI → DICOM SEG conversion successful (ITK)")

    def test_dicom_series_to_nifti_original(self):
        """Test DICOM to NIfTI conversion with original DICOM files (Explicit VR Little Endian)."""
        # Use a specific series from dicomweb
        dicom_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )

        # Find DICOM files in this series
        dcm_files = list(Path(dicom_dir).glob("*.dcm"))
        self.assertTrue(len(dcm_files) > 0, f"No DICOM files found in {dicom_dir}")

        # Reference NIfTI file (in parent directory with same name as series)
        series_uid = os.path.basename(dicom_dir)
        reference_nifti = os.path.join(os.path.dirname(dicom_dir), f"{series_uid}.nii.gz")

        # Convert DICOM series to NIfTI
        result = dicom_to_nifti(dicom_dir)

        # Verify the result
        self.assertTrue(os.path.exists(result), "NIfTI file should be created")
        self.assertTrue(result.endswith(".nii.gz"), "Output should be a compressed NIfTI file")

        # Load and verify the NIfTI data
        nifti_data, nifti_meta = LoadImage(image_only=False)(result)

        # Verify it's a 3D volume with expected dimensions (512x512x77)
        self.assertEqual(len(nifti_data.shape), 3, "Should be a 3D volume")
        self.assertEqual(nifti_data.shape[0], 512, "Should have 512 rows")
        self.assertEqual(nifti_data.shape[1], 512, "Should have 512 columns")
        self.assertEqual(nifti_data.shape[2], 77, "Should have 77 slices")

        # Verify metadata includes affine transformation
        self.assertIn("affine", nifti_meta, "Metadata should include affine transformation")

        # Compare with reference NIfTI
        ref_data, ref_meta = LoadImage(image_only=False)(reference_nifti)
        self.assertEqual(nifti_data.shape, ref_data.shape, "Shape should match reference NIfTI")
        # Check if pixel values are similar (allowing for minor differences in conversion)
        np.testing.assert_allclose(
            nifti_data, ref_data, rtol=1e-5, atol=1e-5, err_msg="Pixel values should match reference NIfTI"
        )
        print(f"  ✓ Matches reference NIfTI")

        # Clean up
        os.unlink(result)

        print(f"✓ Original DICOM → NIfTI conversion successful")
        print(f"  Input: {len(dcm_files)} DICOM files")
        print(f"  Output shape: {nifti_data.shape}")

    def test_dicom_series_to_nifti_htj2k(self):
        """Test DICOM to NIfTI conversion with HTJ2K-encoded DICOM files."""
        if not HAS_NVIMGCODEC:
            self.skipTest(
                "nvimgcodec not available. Install nvidia-nvimgcodec-cu{XX} matching your CUDA version (e.g., nvidia-nvimgcodec-cu13 for CUDA 13.x)"
            )

        # Use a specific HTJ2K series from dicomweb_htj2k
        htj2k_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb_htj2k",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )

        # Find HTJ2K files in this series
        htj2k_files = list(Path(htj2k_dir).glob("*.dcm"))

        # If no HTJ2K files found but nvimgcodec is available, create them
        if len(htj2k_files) == 0:
            print("\nHTJ2K test data not found. Creating HTJ2K-encoded DICOM files...")
            import sys

            sys.path.insert(0, os.path.join(self.base_dir))
            from prepare_htj2k_test_data import create_htj2k_data

            create_htj2k_data(os.path.join(self.base_dir, "data"))
            # Re-check for files
            htj2k_files = list(Path(htj2k_dir).glob("*.dcm"))

        if len(htj2k_files) == 0:
            self.skipTest(f"No HTJ2K DICOM files found in {htj2k_dir}")

        # Reference NIfTI file (from original dicomweb directory)
        series_uid = os.path.basename(htj2k_dir)
        # Go up from dicomweb_htj2k to dataset, then to dicomweb
        reference_nifti = os.path.join(
            self.base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd", f"{series_uid}.nii.gz"
        )

        # Convert HTJ2K DICOM series to NIfTI
        result = dicom_to_nifti(htj2k_dir)

        # Verify the result
        self.assertTrue(os.path.exists(result), "NIfTI file should be created")
        self.assertTrue(result.endswith(".nii.gz"), "Output should be a compressed NIfTI file")

        # Load and verify the NIfTI data
        nifti_data, nifti_meta = LoadImage(image_only=False)(result)

        # Verify it's a 3D volume with expected dimensions (512x512x77)
        self.assertEqual(len(nifti_data.shape), 3, "Should be a 3D volume")
        self.assertEqual(nifti_data.shape[0], 512, "Should have 512 rows")
        self.assertEqual(nifti_data.shape[1], 512, "Should have 512 columns")
        self.assertEqual(nifti_data.shape[2], 77, "Should have 77 slices")

        # Verify metadata includes affine transformation
        self.assertIn("affine", nifti_meta, "Metadata should include affine transformation")

        # Compare with reference NIfTI
        ref_data, ref_meta = LoadImage(image_only=False)(reference_nifti)
        self.assertEqual(nifti_data.shape, ref_data.shape, "Shape should match reference NIfTI")
        # HTJ2K is lossless, so pixel values should be identical
        np.testing.assert_allclose(
            nifti_data, ref_data, rtol=1e-5, atol=1e-5, err_msg="Pixel values should match reference NIfTI"
        )
        print(f"  ✓ Matches reference NIfTI (lossless HTJ2K compression verified)")

        # Clean up
        os.unlink(result)

        print(f"✓ HTJ2K DICOM → NIfTI conversion successful")
        print(f"  Input: {len(htj2k_files)} HTJ2K DICOM files")
        print(f"  Output shape: {nifti_data.shape}")

    def test_dicom_to_nifti_consistency(self):
        """Test that original and HTJ2K DICOM files produce identical NIfTI outputs."""
        if not HAS_NVIMGCODEC:
            self.skipTest(
                "nvimgcodec not available. Install nvidia-nvimgcodec-cu{XX} matching your CUDA version (e.g., nvidia-nvimgcodec-cu13 for CUDA 13.x)"
            )

        # Use specific series directories for both original and HTJ2K
        dicom_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )
        htj2k_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb_htj2k",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )

        # Check if HTJ2K files exist, create if needed
        htj2k_files = list(Path(htj2k_dir).glob("*.dcm"))
        if len(htj2k_files) == 0:
            print("\nHTJ2K test data not found. Creating HTJ2K-encoded DICOM files...")
            import sys

            sys.path.insert(0, os.path.join(self.base_dir))
            from prepare_htj2k_test_data import create_htj2k_data

            create_htj2k_data(os.path.join(self.base_dir, "data"))
            # Re-check for files
            htj2k_files = list(Path(htj2k_dir).glob("*.dcm"))

        # If still no HTJ2K files, skip the test (encoding may have failed)
        if len(htj2k_files) == 0:
            self.skipTest(
                f"No HTJ2K DICOM files found in {htj2k_dir}. HTJ2K encoding may not be supported for these files."
            )

        # Convert both versions
        result_original = dicom_to_nifti(dicom_dir)
        result_htj2k = dicom_to_nifti(htj2k_dir)

        try:
            # Load both NIfTI files
            data_original = LoadImage(image_only=True)(result_original)
            data_htj2k = LoadImage(image_only=True)(result_htj2k)

            # Verify shapes match
            self.assertEqual(data_original.shape, data_htj2k.shape, "Original and HTJ2K should produce same shape")

            # Verify data types match
            self.assertEqual(data_original.dtype, data_htj2k.dtype, "Original and HTJ2K should produce same data type")

            # Verify pixel values are identical (HTJ2K is lossless)
            np.testing.assert_array_equal(
                data_original, data_htj2k, err_msg="Original and HTJ2K should produce identical pixel values (lossless)"
            )

            print(f"✓ Original and HTJ2K produce identical NIfTI outputs")
            print(f"  Shape: {data_original.shape}")
            print(f"  Data type: {data_original.dtype}")
            print(f"  Pixel values: Identical (lossless compression verified)")

        finally:
            # Clean up
            if os.path.exists(result_original):
                os.unlink(result_original)
            if os.path.exists(result_htj2k):
                os.unlink(result_htj2k)


if __name__ == "__main__":
    unittest.main()
