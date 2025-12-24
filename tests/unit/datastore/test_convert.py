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
import pydicom
import SimpleITK as sitk
from monai.transforms import LoadImage

from monailabel.datastore.utils.convert import (
    binary_to_image,
    dicom_seg_to_itk_image,
    dicom_to_nifti,
    nifti_to_dicom_seg,
)


class TestConvert(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    local_dataset = os.path.join(base_dir, "data", "dataset", "local", "spleen")
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")

    # Test data constants
    TEST_SERIES_ID = "1.2.826.0.1.3680043.8.274.1.1.8323329.686549.1629744177.996087"

    # === Utility Methods ===

    def _get_test_paths(self):
        """Get standard DICOM series and label paths."""
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)
        label = os.path.join(
            self.dicom_dataset,
            "labels",
            "final",
            f"{self.TEST_SERIES_ID}.nii.gz",
        )
        return series_dir, label

    def _load_dicom_series(self, series_dir):
        """Load DICOM series and return reference image and dimensions."""
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
        assert len(dicom_names) > 0, f"No DICOM series found in {series_dir}"
        reader.SetFileNames(dicom_names)
        reference_image = reader.Execute()
        return reference_image, reference_image.GetSize()

    def _create_label_file(self, label_array, reference_image):
        """Create a temporary NIfTI label file from array with proper geometry."""
        label_sitk = sitk.GetImageFromArray(label_array)
        label_sitk.CopyInformation(reference_image)
        label_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
        sitk.WriteImage(label_sitk, label_file)
        return label_file

    def _extract_pixels(self, dicom_seg_file, remove_background=True):
        """Extract pixel values from DICOM SEG."""
        result_nifti = dicom_seg_to_itk_image(dicom_seg_file)
        assert os.path.exists(result_nifti), "Failed to convert DICOM SEG back to image"

        try:
            result_img = sitk.ReadImage(result_nifti)
            pixel_array = sitk.GetArrayFromImage(result_img)

            unique_values = np.unique(pixel_array)
            if remove_background:
                unique_values = unique_values[unique_values != 0]

            return pixel_array, unique_values
        finally:
            os.unlink(result_nifti)  # Cleanup temp file

    def _count_segments_in_label(self, label_path):
        """Load label file and count unique segments (excluding background)."""
        label_img = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label_img)
        return len(np.unique(label_array)) - 1

    def _validate_dicom_seg(self, dcm_file, expected_segments=None):
        """Validate basic DICOM SEG attributes and return dataset."""
        dcm = pydicom.dcmread(dcm_file)

        # Accept both Segmentation Storage and Labelmap Segmentation Storage
        valid_sop_classes = [
            "1.2.840.10008.5.1.4.1.1.66.4",  # Segmentation Storage
            "1.2.840.10008.5.1.4.1.1.66.7",  # Labelmap Segmentation Storage
        ]
        assert dcm.SOPClassUID in valid_sop_classes, f"Not a valid DICOM SEG: got {dcm.SOPClassUID}"
        assert len(dcm.SegmentSequence) > 0, "No segments in DICOM SEG"
        assert dcm.NumberOfFrames > 0, "No frames in DICOM SEG"
        assert hasattr(dcm, "PixelData"), "Missing PixelData"

        if expected_segments is not None:
            assert (
                len(dcm.SegmentSequence) == expected_segments
            ), f"Expected {expected_segments} segments, got {len(dcm.SegmentSequence)}"

        return dcm

    def _create_multi_segment_array(self, dims, segment_values, region_type="full_slice"):
        """Create synthetic label array with multiple segments.

        Args:
            dims: Tuple of (width, height, depth)
            segment_values: List of label values (e.g., [1, 5, 10])
            region_type: 'full_slice' (entire slices) or 'regions' (spatial regions)

        Returns:
            numpy array of shape (depth, height, width)
        """
        label_array = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
        z_segment = max(1, dims[2] // len(segment_values))

        for i, value in enumerate(segment_values):
            z_start = i * z_segment
            z_end = (i + 1) * z_segment if i < len(segment_values) - 1 else dims[2]

            if region_type == "full_slice":
                label_array[z_start:z_end, :, :] = value
            elif region_type == "regions":
                # Create non-overlapping regions with safe bounds
                # Enforce minimum sizes (at least 1 pixel) to handle very small images
                box_w = max(1, min(80, dims[0] // (len(segment_values) + 1)))
                box_h = max(1, min(80, dims[1] // 2))

                # Position regions non-overlapping horizontally
                # Enforce minimum spacing to prevent division by zero
                x_spacing = max(1, dims[0] // (len(segment_values) + 1))
                x_center = (i + 1) * x_spacing
                # Clamp x_center to valid range
                x_center = max(0, min(x_center, dims[0] - 1))

                x0 = max(0, x_center - box_w // 2)
                x1 = min(dims[0], x0 + box_w)
                # Ensure at least one column
                x1 = max(x1, x0 + 1)

                # Center vertically
                y_center = dims[1] // 2
                y0 = max(0, y_center - box_h // 2)
                y1 = min(dims[1], y0 + box_h)
                # Ensure at least one row
                y1 = max(y1, y0 + 1)

                label_array[z_start:z_end, y0:y1, x0:x1] = value
            elif region_type == "large_regions":
                # Create large centered regions for robustness (target 100x100)
                # Enforce minimum size to handle very small images
                box_size = max(1, min(100, dims[0] // 2, dims[1] // 2))

                # Center the box in X dimension
                x0 = max(0, dims[0] // 2 - box_size // 2)
                x1 = min(dims[0], x0 + box_size)
                # Ensure at least one column
                x1 = max(x1, x0 + 1)

                # Center the box in Y dimension
                y0 = max(0, dims[1] // 2 - box_size // 2)
                y1 = min(dims[1], y0 + box_size)
                # Ensure at least one row
                y1 = max(y1, y0 + 1)

                label_array[z_start:z_end, y0:y1, x0:x1] = value

        return label_array

    # === Test Methods ===

    def test_dicom_to_nifti(self):
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)
        result = dicom_to_nifti(series_dir)

        assert os.path.exists(result)
        assert result.endswith(".nii.gz")

        # Verify the converted image is valid and has reasonable properties
        result_img = sitk.ReadImage(result)
        result_size = result_img.GetSize()

        # Verify 3D image with reasonable dimensions
        assert len(result_size) == 3, "Should be 3D image"
        assert result_size[0] > 0, "Width must be > 0"
        assert result_size[1] > 0, "Height must be > 0"
        assert result_size[2] > 0, "Depth must be > 0"

        # Verify pixel data is not all zeros
        pixel_array = sitk.GetArrayFromImage(result_img)
        assert not np.all(pixel_array == 0), "Image should not be all zeros"

        # Verify pixel values are in reasonable range for medical imaging (HU units)
        assert pixel_array.min() >= -2048, "Pixel values too low (HU range)"
        assert pixel_array.max() <= 4095, "Pixel values too high (HU range)"

        # Verify spacing is reasonable (not zero, not extreme)
        spacing = result_img.GetSpacing()
        for i, s in enumerate(spacing):
            assert 0.1 < s < 100, f"Spacing[{i}] = {s} is unreasonable"

        os.unlink(result)

    def test_binary_to_image(self):
        reference_image = os.path.join(self.local_dataset, "labels", "final", "spleen_3.nii.gz")

        # Load reference using both methods to get expected values
        ref_img = sitk.ReadImage(reference_image)
        # Geometry validated via spacing checks below; array not needed.
        label = LoadImage(image_only=True)(reference_image)
        label = label.astype(np.uint8)
        original_unique_values = np.unique(label)
        original_nonzero_count = np.count_nonzero(label)
        label = label.flatten(order="F")

        label_bin = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
        label.tofile(label_bin)

        try:
            result = binary_to_image(reference_image, label_bin)
            self.addCleanup(os.unlink, result)
        finally:
            os.unlink(label_bin)

        assert os.path.exists(result)
        assert result.endswith(".nii.gz")

        # Verify the result is valid and readable
        result_img = sitk.ReadImage(result)
        result_array = sitk.GetArrayFromImage(result_img)

        # Verify 3D structure exists
        assert len(result_array.shape) == 3, "Should be 3D image"
        assert result_array.shape[0] > 0, "Depth must be > 0"
        assert result_array.shape[1] > 0, "Height must be > 0"
        assert result_array.shape[2] > 0, "Width must be > 0"

        # Verify geometry matches reference (spacing is preserved)
        result_spacing = result_img.GetSpacing()
        ref_spacing = ref_img.GetSpacing()
        for i in range(3):
            spacing_diff = abs(result_spacing[i] - ref_spacing[i])
            assert spacing_diff < 0.01, f"Spacing mismatch in dim {i}: {ref_spacing[i]} vs {result_spacing[i]}"

        # Verify data content is reasonable (same unique values, similar nonzero count)
        result_unique_values = np.unique(result_array)
        result_nonzero_count = np.count_nonzero(result_array)

        assert set(result_unique_values) == set(
            original_unique_values
        ), f"Unique values changed: {set(original_unique_values)} vs {set(result_unique_values)}"

        # Allow 1% difference in nonzero count due to potential boundary effects
        count_diff_ratio = abs(result_nonzero_count - original_nonzero_count) / max(original_nonzero_count, 1)
        assert (
            count_diff_ratio < 0.01
        ), f"Nonzero count changed significantly: {original_nonzero_count} vs {result_nonzero_count}"

        os.unlink(result)

    def _test_nifti_to_dicom_seg_with_label_info_impl(self, use_itk):
        """Helper: Test NIfTI to DICOM SEG conversion with custom label info."""
        series_dir, label = self._get_test_paths()

        label_info = [
            {
                "name": "Spleen",
                "description": "Spleen organ",
                "color": [255, 0, 0],
                "model_name": "TestModel",
            }
        ]

        result = nifti_to_dicom_seg(series_dir, label, label_info, use_itk=use_itk)
        self.addCleanup(os.unlink, result)

        assert os.path.exists(result)
        dcm = pydicom.dcmread(result)

        # Verify series description
        assert dcm.SeriesDescription == "TestModel"

        # Verify segment metadata is properly set
        # Note: LABELMAP type creates a Background segment with SegmentNumber=0
        assert len(dcm.SegmentSequence) >= 1

        # Find the first real segment (skip Background if present)
        real_segments = [s for s in dcm.SegmentSequence if s.SegmentNumber > 0]
        assert len(real_segments) >= 1, "Should have at least one real segment"
        seg = real_segments[0]

        # Verify segment label from label_info
        assert seg.SegmentLabel == "Spleen", f"Expected 'Spleen', got '{seg.SegmentLabel}'"

        # Note: SegmentDescription is optional in DICOM, may not be present
        # The description is typically in SegmentedPropertyTypeCodeSequence.CodeMeaning

        # Verify algorithm information
        assert seg.SegmentAlgorithmType == "AUTOMATIC"
        assert seg.SegmentAlgorithmName == "MONAILABEL"

        # Verify segment has required code sequences
        assert hasattr(seg, "SegmentedPropertyCategoryCodeSequence")
        assert len(seg.SegmentedPropertyCategoryCodeSequence) > 0
        assert hasattr(seg, "SegmentedPropertyTypeCodeSequence")
        assert len(seg.SegmentedPropertyTypeCodeSequence) > 0

        # Verify pixel data exists
        assert hasattr(dcm, "PixelData")
        assert len(dcm.PixelData) > 0

        # Verify frame count
        assert hasattr(dcm, "NumberOfFrames")
        assert dcm.NumberOfFrames > 0

    def test_nifti_to_dicom_seg_with_label_info_highdicom(self):
        """Test label info conversion using highdicom implementation."""
        self._test_nifti_to_dicom_seg_with_label_info_impl(use_itk=False)

    def test_nifti_to_dicom_seg_with_label_info_itk(self):
        """Test label info conversion using ITK implementation."""
        self._test_nifti_to_dicom_seg_with_label_info_impl(use_itk=True)

    def _test_segment_number_mapping_impl(self, use_itk):
        """Helper: Test that non-sequential label values are correctly mapped to sequential segment numbers."""
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)

        # Load DICOM series to get proper dimensions
        reference_image, dims = self._load_dicom_series(series_dir)

        # Create synthetic label with non-sequential values (1, 5, 10)
        label_array = self._create_multi_segment_array(dims, [1, 5, 10], region_type="large_regions")
        label_file = self._create_label_file(label_array, reference_image)

        # Define label info for all three segments
        label_info = [
            {"name": "Segment1", "description": "First segment", "color": [255, 0, 0]},
            {"name": "Segment5", "description": "Second segment", "color": [0, 255, 0]},
            {"name": "Segment10", "description": "Third segment", "color": [0, 0, 255]},
        ]

        # Convert to DICOM SEG
        result = nifti_to_dicom_seg(series_dir, label_file, label_info, use_itk=use_itk)

        assert os.path.exists(result)

        # Read back and verify
        dcm = pydicom.dcmread(result)

        # Filter out Background segment (SegmentNumber=0) created by LABELMAP type
        real_segments = [s for s in dcm.SegmentSequence if s.SegmentNumber > 0]

        # Verify we have 3 real segments
        assert len(real_segments) == 3, f"Expected 3 real segments, got {len(real_segments)}"

        # Verify segment numbers are sequential (1, 2, 3)
        # This is the main bug we fixed - non-sequential labels (1, 5, 10)
        # should be remapped to sequential segment numbers (1, 2, 3)
        segment_numbers = [seg_item.SegmentNumber for seg_item in real_segments]
        assert segment_numbers == [1, 2, 3], f"Expected [1, 2, 3], got {segment_numbers}"

        # Verify segment labels match our input
        segment_labels = [seg_item.SegmentLabel for seg_item in real_segments]
        assert segment_labels == ["Segment1", "Segment5", "Segment10"], f"Expected correct labels, got {segment_labels}"

        # Verify it's a valid DICOM SEG (accept both Segmentation Storage and Labelmap Segmentation Storage)
        assert dcm.SOPClassUID in ["1.2.840.10008.5.1.4.1.1.66.4", "1.2.840.10008.5.1.4.1.1.66.7"]

        # Verify pixel data
        pixel_array, unique_values = self._extract_pixels(result)

        # Must have exactly 3 segments with values 1, 2, 3
        assert (
            len(unique_values) == 3
        ), f"Expected exactly 3 segments in pixel data, found {len(unique_values)}: {list(unique_values)}"

        # Verify segments are exactly {1, 2, 3}, not the original {1, 5, 10}
        assert set(unique_values) == {
            1,
            2,
            3,
        }, f"Pixel values must be {{1,2,3}} (remapped from {{1,5,10}}), got {set(unique_values)}"

        # SPATIAL VERIFICATION: Verify each segment has substantial voxels
        # Compute expected voxels based on actual test geometry
        # "large_regions" creates box_size x box_size x z_segment regions
        num_segments = 3
        box_size = max(1, min(100, dims[0] // 2, dims[1] // 2))
        z_segment = max(1, dims[2] // num_segments)

        # For the last segment, it gets all remaining slices
        z_slices_per_segment = [z_segment, z_segment, dims[2] - 2 * z_segment]

        # Compute expected voxels per segment (box_size * box_size * z_slices)
        expected_voxels_per_segment = [box_size * box_size * z for z in z_slices_per_segment]

        # Use 80% tolerance to account for compression and round-trip losses
        min_expected_per_segment = [int(expected * 0.8) for expected in expected_voxels_per_segment]
        expected_total = int(sum(expected_voxels_per_segment) * 0.8)

        count_per_segment = {1: np.sum(pixel_array == 1), 2: np.sum(pixel_array == 2), 3: np.sum(pixel_array == 3)}

        # Verify each segment has expected voxels (with tolerance)
        for i, (seg_num, count) in enumerate(count_per_segment.items()):
            min_expected = min_expected_per_segment[i]
            assert (
                count >= min_expected
            ), f"Segment {seg_num} has too few voxels: {count} < {min_expected} (box_size={box_size}, z_slices={z_slices_per_segment[i]})"

        # Verify total matches expected
        total_nonzero = sum(count_per_segment.values())
        assert (
            total_nonzero >= expected_total
        ), f"Total segmentation voxels too low: {total_nonzero} (expected >= {expected_total}, box_size={box_size}, dims={dims})"

        # Verify frames exist in DICOM SEG
        assert dcm.NumberOfFrames > 0, "No frames in segmentation"

        # Cleanup
        os.unlink(label_file)
        os.unlink(result)

    def test_segment_number_mapping_highdicom(self):
        """Test non-sequential label remapping using highdicom implementation."""
        self._test_segment_number_mapping_impl(use_itk=False)

    def test_segment_number_mapping_itk(self):
        """Test non-sequential label remapping using ITK implementation."""
        self._test_segment_number_mapping_impl(use_itk=True)

    def _test_round_trip_impl(self, use_itk):
        """Helper: Test NIfTI → DICOM SEG → NIfTI preserves data accurately."""
        series_dir, label = self._get_test_paths()

        label_info = [
            {"name": "Spleen", "color": [255, 0, 0]},
        ]

        # Load original for comparison
        original_sitk = sitk.ReadImage(label)
        original_array = sitk.GetArrayFromImage(original_sitk)
        original_spacing = original_sitk.GetSpacing()
        original_size = original_sitk.GetSize()

        # Convert to DICOM SEG
        dicom_seg_file = nifti_to_dicom_seg(series_dir, label, label_info, use_itk=use_itk)
        assert os.path.exists(dicom_seg_file)

        # Convert back to NIfTI
        result_nifti = dicom_seg_to_itk_image(dicom_seg_file)
        assert os.path.exists(result_nifti)

        # Load result
        result_sitk = sitk.ReadImage(result_nifti)
        result_array = sitk.GetArrayFromImage(result_sitk)
        result_spacing = result_sitk.GetSpacing()
        result_size = result_sitk.GetSize()

        # Verify dimensions are preserved
        # NOTE: ITK/dcmqi only stores non-empty slices, so Z dimension may differ
        if use_itk:
            # For ITK, only check X/Y dimensions match
            assert result_size[0] == original_size[0], f"X dimension changed: {original_size[0]} → {result_size[0]}"
            assert result_size[1] == original_size[1], f"Y dimension changed: {original_size[1]} → {result_size[1]}"
            # Z dimension will be smaller (only non-empty slices)
            assert (
                result_size[2] <= original_size[2]
            ), f"Z dimension should be ≤ original: {original_size[2]} → {result_size[2]}"
        else:
            # For highdicom, expect exact dimension match
            assert result_size == original_size, f"Dimensions changed: {original_size} → {result_size}"

        # Verify geometry is preserved (with tolerance for floating point)
        for i in range(3):
            spacing_diff = abs(original_spacing[i] - result_spacing[i])
            assert spacing_diff < 0.01, f"Spacing changed in dimension {i}: {original_spacing[i]} → {result_spacing[i]}"

        # Note: Origin may not be preserved through DICOM SEG conversion
        # This is expected due to DICOM coordinate system transformations
        # We primarily care that dimensions, spacing, and data are correct

        # Verify unique values are preserved
        original_unique = sorted(np.unique(original_array))
        result_unique = sorted(np.unique(result_array))
        assert original_unique == result_unique, f"Unique values mismatch: {original_unique} vs {result_unique}"

        # Verify label counts are similar (within 3% due to potential compression/resampling)
        for label_value in original_unique:
            if label_value == 0:
                continue
            orig_count = np.sum(original_array == label_value)
            result_count = np.sum(result_array == label_value)

            if orig_count > 0:
                ratio = result_count / orig_count
                assert (
                    0.97 <= ratio <= 1.03
                ), f"Label {label_value} count changed by >3%: {orig_count} → {result_count} (ratio: {ratio:.2f})"

        # Verify result is not empty
        assert np.any(result_array != 0), "Result segmentation is all zeros"

        # Cleanup
        os.unlink(dicom_seg_file)
        os.unlink(result_nifti)

    def test_round_trip_highdicom(self):
        """Test round-trip conversion using highdicom implementation."""
        self._test_round_trip_impl(use_itk=False)

    def test_round_trip_itk(self):
        """Test round-trip conversion using ITK implementation."""
        self._test_round_trip_impl(use_itk=True)

    def _test_empty_label_impl(self, use_itk):
        """Helper: Test handling of empty label files."""
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)

        # Load DICOM series to get reference image
        reference_image, _ = self._load_dicom_series(series_dir)

        # Create an empty label (all zeros) with matching dimensions
        label_sitk = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
        label_sitk.CopyInformation(reference_image)
        label_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
        sitk.WriteImage(label_sitk, label_file)

        # Should return empty string for empty label
        result = nifti_to_dicom_seg(series_dir, label_file, None, use_itk=use_itk)
        assert result == ""

        # Cleanup
        os.unlink(label_file)

    def test_empty_label_highdicom(self):
        """Test empty label handling using highdicom implementation."""
        self._test_empty_label_impl(use_itk=False)

    def test_empty_label_itk(self):
        """Test empty label handling using ITK implementation."""
        self._test_empty_label_impl(use_itk=True)

    def _test_missing_label_info_impl(self, use_itk):
        """Helper: Test that conversion works with missing/incomplete label_info and applies defaults correctly."""
        series_dir, label = self._get_test_paths()

        # Count expected segments in label
        expected_segments = self._count_segments_in_label(label)

        # Convert with None label_info
        result = nifti_to_dicom_seg(series_dir, label, None, use_itk=use_itk)
        assert os.path.exists(result)

        # Verify default values were used for all segments
        dcm = pydicom.dcmread(result)

        # Filter out Background segment (SegmentNumber=0) for LABELMAP type
        real_segments = [s for s in dcm.SegmentSequence if s.SegmentNumber > 0]

        assert (
            len(real_segments) == expected_segments
        ), f"Expected {expected_segments} real segments, got {len(real_segments)}"

        # Verify each real segment has default naming
        for i, seg in enumerate(real_segments, start=1):
            assert seg.SegmentLabel == f"Segment_{i}", f"Expected default label 'Segment_{i}', got '{seg.SegmentLabel}'"

            # Verify default algorithm info
            assert seg.SegmentAlgorithmType == "AUTOMATIC"
            assert seg.SegmentAlgorithmName == "MONAILABEL"

            # Verify has default code sequences
            assert hasattr(seg, "SegmentedPropertyCategoryCodeSequence")
            assert len(seg.SegmentedPropertyCategoryCodeSequence) > 0

        # Cleanup
        os.unlink(result)

    def test_missing_label_info_highdicom(self):
        """Test missing label_info handling using highdicom implementation."""
        self._test_missing_label_info_impl(use_itk=False)

    def test_missing_label_info_itk(self):
        """Test missing label_info handling using ITK implementation."""
        self._test_missing_label_info_impl(use_itk=True)

    def test_dicom_seg_to_itk_image(self):
        """Test DICOM SEG to NIfTI/NRRD conversion."""
        series_dir, label = self._get_test_paths()

        # Count expected segments in original label
        expected_segments = self._count_segments_in_label(label)

        # First create a DICOM SEG
        dicom_seg_file = nifti_to_dicom_seg(series_dir, label, None, use_itk=False)
        self.addCleanup(os.unlink, dicom_seg_file)
        assert os.path.exists(dicom_seg_file)

        # Convert to ITK image
        result = dicom_seg_to_itk_image(dicom_seg_file)
        self.addCleanup(os.unlink, result)
        assert os.path.exists(result)
        assert result.endswith(".seg.nrrd")

        # Verify it's readable and has correct structure
        result_img = sitk.ReadImage(result)
        result_array = sitk.GetArrayFromImage(result_img)

        # Verify 3D structure
        assert result_img.GetSize()[0] > 0, "Width must be > 0"
        assert result_img.GetSize()[1] > 0, "Height must be > 0"
        assert result_img.GetSize()[2] > 0, "Depth must be > 0"

        # Verify pixel data contains segments
        unique_values = np.unique(result_array)
        unique_values = unique_values[unique_values != 0]

        assert (
            len(unique_values) == expected_segments
        ), f"Expected {expected_segments} segments, found {len(unique_values)}"

        # Verify result is not empty
        assert np.any(result_array != 0), "Result is all zeros"

        # Verify spacing is reasonable
        spacing = result_img.GetSpacing()
        for i, s in enumerate(spacing):
            assert 0.1 < s < 100, f"Spacing[{i}] = {s} is unreasonable"

    def test_custom_tags(self):
        """Test that custom DICOM tags are properly added."""
        series_dir, label = self._get_test_paths()

        custom_tags = {"ContentCreatorName": "TestUser", "ClinicalTrialSeriesID": "TRIAL123"}

        result = nifti_to_dicom_seg(series_dir, label, None, use_itk=False, custom_tags=custom_tags)
        self.addCleanup(os.unlink, result)
        assert os.path.exists(result)

        # Verify custom tags
        dcm = pydicom.dcmread(result)
        assert dcm.ContentCreatorName == "TestUser"
        assert dcm.ClinicalTrialSeriesID == "TRIAL123"

    def _test_multiple_segments_with_different_properties_impl(self, use_itk):
        """Helper: Test multiple segments each with unique names, colors, and descriptions."""
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)

        # Load DICOM series and create label with 3 different segments
        reference_image, dims = self._load_dicom_series(series_dir)
        label_array = self._create_multi_segment_array(dims, [1, 2, 3], region_type="regions")
        label_file = self._create_label_file(label_array, reference_image)
        self.addCleanup(os.unlink, label_file)

        # Define distinct properties for each segment
        label_info = [
            {
                "name": "Liver",
                "description": "Liver structure",
                "color": [255, 0, 0],
            },
            {
                "name": "Spleen",
                "description": "Spleen structure",
                "color": [0, 255, 0],
            },
            {
                "name": "Kidney",
                "description": "Kidney structure",
                "color": [0, 0, 255],
            },
        ]

        # Convert to DICOM SEG
        result = nifti_to_dicom_seg(series_dir, label_file, label_info, use_itk=use_itk)
        self.addCleanup(os.unlink, result)
        assert os.path.exists(result)

        # Verify all segments have correct properties
        dcm = pydicom.dcmread(result)

        # Filter out Background segment (SegmentNumber=0) for LABELMAP type
        real_segments = [s for s in dcm.SegmentSequence if s.SegmentNumber > 0]
        assert len(real_segments) == 3

        # Verify each real segment's properties
        for i, expected_info in enumerate(label_info):
            seg = real_segments[i]
            assert seg.SegmentNumber == i + 1
            assert (
                seg.SegmentLabel == expected_info["name"]
            ), f"Segment {i + 1}: expected '{expected_info['name']}', got '{seg.SegmentLabel}'"

            # Note: SegmentDescription is optional in DICOM
            # Verify the segment has required code sequences instead
            assert hasattr(seg, "SegmentedPropertyCategoryCodeSequence")
            assert hasattr(seg, "SegmentedPropertyTypeCodeSequence")

        # Verify all 3 real segments are present in metadata
        segments_in_metadata = [seg.SegmentNumber for seg in real_segments]

        assert (
            len(segments_in_metadata) == 3
        ), f"Expected 3 real segments in metadata, found {len(segments_in_metadata)}"
        assert set(segments_in_metadata) == {1, 2, 3}, f"Expected segments {{1,2,3}}, found {set(segments_in_metadata)}"

        # Verify frames exist for the segments
        assert dcm.NumberOfFrames > 0, "No frames in DICOM SEG"

        # Verify pixel data contains all 3 segments via round-trip
        result_nifti = dicom_seg_to_itk_image(result)
        self.addCleanup(os.unlink, result_nifti)
        result_img = sitk.ReadImage(result_nifti)
        pixel_array = sitk.GetArrayFromImage(result_img)

        unique_in_pixels = np.unique(pixel_array)
        unique_in_pixels = unique_in_pixels[unique_in_pixels != 0]

        assert len(unique_in_pixels) == 3, f"Expected 3 segments in pixel data, found {len(unique_in_pixels)}"
        assert set(unique_in_pixels) == {1, 2, 3}, f"Expected pixel values {{1,2,3}}, found {set(unique_in_pixels)}"

    def test_multiple_segments_with_different_properties_highdicom(self):
        """Test multiple segments with unique properties using highdicom implementation."""
        self._test_multiple_segments_with_different_properties_impl(use_itk=False)

    def test_multiple_segments_with_different_properties_itk(self):
        """Test multiple segments with unique properties using ITK implementation."""
        self._test_multiple_segments_with_different_properties_impl(use_itk=True)

    def _test_large_label_values_impl(self, use_itk):
        """Helper: Test that large label values (100, 200, 255) are correctly remapped to sequential (1, 2, 3)."""
        series_dir = os.path.join(self.dicom_dataset, self.TEST_SERIES_ID)

        # Load DICOM series and create label with large values: 100, 200, 255
        reference_image, dims = self._load_dicom_series(series_dir)
        label_array = self._create_multi_segment_array(dims, [100, 200, 255], region_type="full_slice")
        label_file = self._create_label_file(label_array, reference_image)

        label_info = [
            {"name": "Segment100"},
            {"name": "Segment200"},
            {"name": "Segment255"},
        ]

        # Convert to DICOM SEG
        result = nifti_to_dicom_seg(series_dir, label_file, label_info, use_itk=use_itk)
        assert os.path.exists(result)

        # Verify segment numbers are sequential 1, 2, 3 (not 100, 200, 255)
        dcm = pydicom.dcmread(result)

        # Filter out Background segment (SegmentNumber=0) for LABELMAP type
        real_segments = [s for s in dcm.SegmentSequence if s.SegmentNumber > 0]

        assert len(real_segments) == 3
        segment_numbers = [seg.SegmentNumber for seg in real_segments]
        assert segment_numbers == [1, 2, 3], f"Large values not remapped: expected [1, 2, 3], got {segment_numbers}"

        # Verify labels are preserved
        segment_labels = [seg.SegmentLabel for seg in real_segments]
        assert segment_labels == ["Segment100", "Segment200", "Segment255"]

        # Verify segment numbers are remapped to 1, 2, 3 (not 100, 200, 255)
        segments_in_metadata = [seg.SegmentNumber for seg in real_segments]

        assert (
            len(segments_in_metadata) == 3
        ), f"Expected 3 real segments in metadata, found {len(segments_in_metadata)}"
        assert set(segments_in_metadata) == {
            1,
            2,
            3,
        }, f"Segment numbers must be {{1,2,3}} not {{100,200,255}}, got {set(segments_in_metadata)}"

        # Verify frames exist
        assert dcm.NumberOfFrames > 0, "No frames in DICOM SEG"

        # Verify pixel data also contains remapped values via round-trip
        # With full-slice regions, all segments should survive round-trip
        result_nifti = dicom_seg_to_itk_image(result)
        result_img = sitk.ReadImage(result_nifti)
        pixel_array = sitk.GetArrayFromImage(result_img)

        unique_in_pixels = np.unique(pixel_array)
        unique_in_pixels = unique_in_pixels[unique_in_pixels != 0]

        assert len(unique_in_pixels) == 3, f"Expected 3 segments in pixel data, found {len(unique_in_pixels)}"
        assert set(unique_in_pixels) == {
            1,
            2,
            3,
        }, f"Pixel values must be {{1,2,3}} not {{100,200,255}}, found {set(unique_in_pixels)}"

        os.unlink(result_nifti)

        # Cleanup
        os.unlink(label_file)
        os.unlink(result)

    def test_large_label_values_highdicom(self):
        """Test large label value remapping using highdicom implementation."""
        self._test_large_label_values_impl(use_itk=False)

    def test_large_label_values_itk(self):
        """Test large label value remapping using ITK implementation."""
        self._test_large_label_values_impl(use_itk=True)

    def test_invalid_series_directory(self):
        """Test handling of non-existent DICOM directory."""
        invalid_dir = "/nonexistent/path/to/dicom"
        label = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name

        # Create a minimal valid label file
        label_img = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        label_array = sitk.GetArrayFromImage(label_img)
        label_array[5, 5, 5] = 1
        label_img = sitk.GetImageFromArray(label_array)
        sitk.WriteImage(label_img, label)

        try:
            result = nifti_to_dicom_seg(invalid_dir, label, None, use_itk=False)
            assert result == ""
        finally:
            os.unlink(label)

    def test_dicom_seg_metadata_completeness(self):
        """Test that generated DICOM SEG has all required metadata."""
        series_dir, label = self._get_test_paths()

        result = nifti_to_dicom_seg(series_dir, label, None, use_itk=False)
        assert os.path.exists(result)

        dcm = pydicom.dcmread(result)

        # Verify required DICOM SEG attributes exist
        required_attrs = [
            "SOPClassUID",
            "SOPInstanceUID",
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "SeriesNumber",
            "InstanceNumber",
            "SegmentSequence",
            "PixelData",
            "Rows",
            "Columns",
            "NumberOfFrames",
        ]

        for attr in required_attrs:
            assert hasattr(dcm, attr), f"Missing required attribute: {attr}"

        # Verify SOP Class is Segmentation Storage (or Labelmap Segmentation Storage for LABELMAP type)
        assert dcm.SOPClassUID in [
            "1.2.840.10008.5.1.4.1.1.66.4",  # Segmentation Storage
            "1.2.840.10008.5.1.4.1.1.66.7",  # Labelmap Segmentation Storage
        ]

        # Verify Modality is SEG
        assert dcm.Modality == "SEG"

        # Verify SegmentSequence is not empty
        assert len(dcm.SegmentSequence) > 0

        # Verify each segment has required attributes
        for seg in dcm.SegmentSequence:
            seg_attrs = ["SegmentNumber", "SegmentLabel", "SegmentAlgorithmType"]
            for attr in seg_attrs:
                assert hasattr(seg, attr), f"Segment missing required attribute: {attr}"

        os.unlink(result)


if __name__ == "__main__":
    unittest.main()
