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

        # Load with NvDicomReader (default depth_last=True matches NIfTI W,H,D layout)
        reader = NvDicomReader()
        img_obj = reader.read(self.original_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify shape (should be W, H, D with depth_last=True, the default)
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

        # Load with NvDicomReader (default depth_last=True matches NIfTI W,H,D layout)
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.htj2k_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify shape (should be W, H, D with depth_last=True, the default)
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

        # Load original series (default depth_last=True for W,H,D layout)
        reader_original = NvDicomReader(use_nvimgcodec=False)  # Force pydicom for original
        img_obj_orig = reader_original.read(self.original_series_dir)
        volume_orig, metadata_orig = reader_original.get_data(img_obj_orig)

        # Load HTJ2K series with nvImageCodec (default depth_last=True for W,H,D layout)
        reader_htj2k = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
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

        reader = NvDicomReader()  # default depth_last=True
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

    def test_nvdicomreader_depth_last(self):
        """Test NvDicomReader with depth_last option (ITK-style vs NumPy-style layout)."""
        if not self._check_test_data(self.original_series_dir):
            self.skipTest(f"Original DICOM test data not found at {self.original_series_dir}")

        # NumPy-style: depth_last=False -> (depth, height, width)
        reader_numpy = NvDicomReader(depth_last=False)
        img_obj_numpy = reader_numpy.read(self.original_series_dir)
        volume_numpy, _ = reader_numpy.get_data(img_obj_numpy)

        # ITK-style (default): depth_last=True -> (width, height, depth)
        reader_itk = NvDicomReader(depth_last=True)
        img_obj_itk = reader_itk.read(self.original_series_dir)
        volume_itk, _ = reader_itk.get_data(img_obj_itk)

        # Verify shapes are transposed correctly
        self.assertEqual(volume_numpy.shape, (77, 512, 512))
        self.assertEqual(volume_itk.shape, (512, 512, 77))

        # Verify data is the same (just transposed)
        np.testing.assert_allclose(
            volume_numpy.transpose(2, 1, 0),
            volume_itk,
            rtol=1e-6,
            err_msg="depth_last should produce transposed volume",
        )

        print(f"✓ NvDicomReader depth_last test passed")


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

        # Load with batch decode enabled (default depth_last=True gives W,H,D layout)
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.htj2k_series_dir)
        volume, metadata = reader.get_data(img_obj)

        # Verify successful decode
        self.assertIsNotNone(volume, "Volume should be decoded successfully")
        # With depth_last=True (default), shape is (W, H, D), so depth is at index 2
        self.assertEqual(volume.shape[2], len(htj2k_files), f"Volume should have {len(htj2k_files)} slices")

        print(f"✓ Batch decode optimization test passed ({len(htj2k_files)} slices)")


@unittest.skipIf(not HAS_NVDICOMREADER, "NvDicomReader not available")
@unittest.skipIf(not HAS_PYDICOM, "pydicom not available")
class TestNvDicomReaderMultiFrame(unittest.TestCase):
    """Test suite for NvDicomReader with multi-frame DICOM files."""

    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Single-frame series paths
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")
    htj2k_single_base = os.path.join(base_dir, "data", "dataset", "dicomweb_htj2k", "e7567e0a064f0c334226a0658de23afd")

    # Multi-frame paths (organized by study UID directly)
    htj2k_multiframe_base = os.path.join(base_dir, "data", "dataset", "dicomweb_htj2k_multiframe")

    # Test series UIDs
    test_study_uid = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656706"
    test_series_uid = "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721"

    def setUp(self):
        """Set up test fixtures."""
        self.original_series_dir = os.path.join(self.dicom_dataset, self.test_series_uid)
        self.htj2k_series_dir = os.path.join(self.htj2k_single_base, self.test_series_uid)
        self.multiframe_file = os.path.join(
            self.htj2k_multiframe_base, self.test_study_uid, f"{self.test_series_uid}.dcm"
        )

    def _check_multiframe_data(self):
        """Check if multi-frame test data exists."""
        if not os.path.exists(self.multiframe_file):
            return False
        return True

    def _check_single_frame_data(self):
        """Check if single-frame test data exists."""
        if not os.path.exists(self.original_series_dir):
            return False
        dcm_files = list(Path(self.original_series_dir).glob("*.dcm"))
        if len(dcm_files) == 0:
            return False
        return True

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available for HTJ2K decoding")
    def test_multiframe_basic_read(self):
        """Test that multi-frame DICOM can be read successfully."""
        if not self._check_multiframe_data():
            self.skipTest(f"Multi-frame DICOM not found at {self.multiframe_file}")

        # Read multi-frame DICOM
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.multiframe_file)
        volume, metadata = reader.get_data(img_obj)

        # Convert to numpy if cupy array
        if hasattr(volume, "__cuda_array_interface__"):
            import cupy as cp

            volume = cp.asnumpy(volume)

        # Verify shape (should be W, H, D with depth_last=True)
        self.assertEqual(len(volume.shape), 3, f"Volume should be 3D, got shape {volume.shape}")
        self.assertEqual(volume.shape[2], 77, f"Expected 77 slices, got {volume.shape[2]}")

        # Verify metadata
        self.assertIn("affine", metadata, "Metadata should contain affine matrix")
        self.assertIn("spacing", metadata, "Metadata should contain spacing")
        self.assertIn("ImagePositionPatient", metadata, "Metadata should contain ImagePositionPatient")

        print(f"✓ Multi-frame basic read test passed - shape: {volume.shape}")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available for HTJ2K decoding")
    def test_multiframe_vs_singleframe_consistency(self):
        """Test that multi-frame DICOM produces identical results to single-frame series."""
        if not self._check_multiframe_data():
            self.skipTest(f"Multi-frame DICOM not found at {self.multiframe_file}")

        if not self._check_single_frame_data():
            self.skipTest(f"Single-frame series not found at {self.original_series_dir}")

        # Read single-frame series
        reader_single = NvDicomReader(use_nvimgcodec=False, prefer_gpu_output=False)
        img_obj_single = reader_single.read(self.original_series_dir)
        volume_single, metadata_single = reader_single.get_data(img_obj_single)

        # Read multi-frame DICOM
        reader_multi = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj_multi = reader_multi.read(self.multiframe_file)
        volume_multi, metadata_multi = reader_multi.get_data(img_obj_multi)

        # Convert to numpy if needed
        if hasattr(volume_single, "__cuda_array_interface__"):
            import cupy as cp

            volume_single = cp.asnumpy(volume_single)
        if hasattr(volume_multi, "__cuda_array_interface__"):
            import cupy as cp

            volume_multi = cp.asnumpy(volume_multi)

        # Verify shapes match
        self.assertEqual(
            volume_single.shape,
            volume_multi.shape,
            f"Single-frame and multi-frame volumes should have same shape. Single: {volume_single.shape}, Multi: {volume_multi.shape}",
        )

        # Compare pixel data (HTJ2K lossless should be identical)
        np.testing.assert_allclose(
            volume_single,
            volume_multi,
            rtol=1e-5,
            atol=1e-3,
            err_msg="Multi-frame DICOM pixel data differs from single-frame series",
        )

        # Compare spacing
        np.testing.assert_allclose(
            metadata_single["spacing"], metadata_multi["spacing"], rtol=1e-6, err_msg="Spacing should be identical"
        )

        # Compare affine matrices
        np.testing.assert_allclose(
            metadata_single["affine"],
            metadata_multi["affine"],
            rtol=1e-6,
            atol=1e-3,
            err_msg="Affine matrices should be identical",
        )

        print(f"✓ Multi-frame vs single-frame consistency test passed")
        print(f"  Shape: {volume_multi.shape}")
        print(f"  Spacing: {metadata_multi['spacing']}")
        print(f"  Affine origin: {metadata_multi['affine'][:3, 3]}")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available")
    def test_multiframe_per_frame_metadata(self):
        """Test that per-frame metadata is correctly extracted from PerFrameFunctionalGroupsSequence."""
        if not self._check_multiframe_data():
            self.skipTest(f"Multi-frame DICOM not found at {self.multiframe_file}")

        # Read the DICOM file directly with pydicom to check PerFrameFunctionalGroupsSequence
        ds = pydicom.dcmread(self.multiframe_file)

        # Verify it's actually multi-frame
        self.assertTrue(hasattr(ds, "NumberOfFrames"), "Should have NumberOfFrames attribute")
        self.assertGreater(ds.NumberOfFrames, 1, "Should have multiple frames")

        # Verify PerFrameFunctionalGroupsSequence exists
        self.assertTrue(
            hasattr(ds, "PerFrameFunctionalGroupsSequence"),
            "Multi-frame DICOM should have PerFrameFunctionalGroupsSequence",
        )

        # Verify first frame has PlanePositionSequence
        first_frame = ds.PerFrameFunctionalGroupsSequence[0]
        self.assertTrue(hasattr(first_frame, "PlanePositionSequence"), "First frame should have PlanePositionSequence")

        first_pos = first_frame.PlanePositionSequence[0].ImagePositionPatient
        self.assertEqual(len(first_pos), 3, "ImagePositionPatient should have 3 coordinates")

        # Now read with NvDicomReader and verify metadata is extracted
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.multiframe_file)
        volume, metadata = reader.get_data(img_obj)

        # Verify ImagePositionPatient was extracted from per-frame metadata
        self.assertIn("ImagePositionPatient", metadata, "Should have ImagePositionPatient in metadata")

        extracted_pos = metadata["ImagePositionPatient"]
        self.assertEqual(len(extracted_pos), 3, "Extracted ImagePositionPatient should have 3 coordinates")

        # Verify it matches the first frame position
        np.testing.assert_allclose(
            extracted_pos, first_pos, rtol=1e-6, err_msg="Extracted ImagePositionPatient should match first frame"
        )

        print(f"✓ Multi-frame per-frame metadata test passed")
        print(f"  NumberOfFrames: {ds.NumberOfFrames}")
        print(f"  First frame ImagePositionPatient: {first_pos}")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available")
    def test_multiframe_affine_origin(self):
        """Test that affine matrix origin is correctly extracted from multi-frame per-frame metadata."""
        if not self._check_multiframe_data():
            self.skipTest(f"Multi-frame DICOM not found at {self.multiframe_file}")

        # Read with pydicom to get expected origin
        ds = pydicom.dcmread(self.multiframe_file)
        first_frame = ds.PerFrameFunctionalGroupsSequence[0]
        expected_origin = np.array(first_frame.PlanePositionSequence[0].ImagePositionPatient)

        # Read with NvDicomReader
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False, affine_lps_to_ras=True)
        img_obj = reader.read(self.multiframe_file)
        volume, metadata = reader.get_data(img_obj)

        # Extract origin from affine matrix (after LPS->RAS conversion)
        # RAS affine has origin in last column, first 3 rows
        affine_origin_ras = metadata["affine"][:3, 3]

        # Convert expected_origin from LPS to RAS for comparison
        # LPS to RAS: negate X and Y
        expected_origin_ras = expected_origin.copy()
        expected_origin_ras[0] = -expected_origin_ras[0]
        expected_origin_ras[1] = -expected_origin_ras[1]

        # Verify affine origin matches the first frame's ImagePositionPatient (in RAS)
        np.testing.assert_allclose(
            affine_origin_ras,
            expected_origin_ras,
            rtol=1e-6,
            atol=1e-3,
            err_msg=f"Affine origin should match first frame ImagePositionPatient. Got {affine_origin_ras}, expected {expected_origin_ras}",
        )

        print(f"✓ Multi-frame affine origin test passed")
        print(f"  ImagePositionPatient (LPS): {expected_origin}")
        print(f"  Affine origin (RAS): {affine_origin_ras}")

    @unittest.skipIf(not HAS_NVIMGCODEC, "nvimgcodec not available")
    def test_multiframe_slice_spacing(self):
        """Test that slice spacing is correctly calculated for multi-frame DICOMs."""
        if not self._check_multiframe_data():
            self.skipTest(f"Multi-frame DICOM not found at {self.multiframe_file}")

        # Read with pydicom to get first and last frame positions
        ds = pydicom.dcmread(self.multiframe_file)
        num_frames = ds.NumberOfFrames

        first_frame = ds.PerFrameFunctionalGroupsSequence[0]
        last_frame = ds.PerFrameFunctionalGroupsSequence[num_frames - 1]

        first_pos = np.array(first_frame.PlanePositionSequence[0].ImagePositionPatient)
        last_pos = np.array(last_frame.PlanePositionSequence[0].ImagePositionPatient)

        # Calculate expected slice spacing
        # Distance between first and last divided by (number of slices - 1)
        distance = np.linalg.norm(last_pos - first_pos)
        expected_spacing = distance / (num_frames - 1)

        # Read with NvDicomReader
        reader = NvDicomReader(use_nvimgcodec=True, prefer_gpu_output=False)
        img_obj = reader.read(self.multiframe_file)
        volume, metadata = reader.get_data(img_obj)

        # Get slice spacing (Z spacing, index 2)
        slice_spacing = metadata["spacing"][2]

        # Verify it matches expected
        self.assertAlmostEqual(
            slice_spacing,
            expected_spacing,
            delta=0.1,
            msg=f"Slice spacing should be ~{expected_spacing:.2f}mm, got {slice_spacing:.2f}mm",
        )

        print(f"✓ Multi-frame slice spacing test passed")
        print(f"  Number of frames: {num_frames}")
        print(f"  First position: {first_pos}")
        print(f"  Last position: {last_pos}")
        print(f"  Calculated spacing: {slice_spacing:.4f}mm (expected: {expected_spacing:.4f}mm)")


if __name__ == "__main__":
    unittest.main()
