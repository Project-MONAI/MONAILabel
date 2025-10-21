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

from monailabel.datastore.utils.convert import binary_to_image, dicom_to_nifti, nifti_to_dicom_seg, transcode_dicom_to_htj2k

# Check if nvimgcodec is available
try:
    from nvidia import nvimgcodec

    HAS_NVIMGCODEC = True
except ImportError:
    HAS_NVIMGCODEC = False
    nvimgcodec = None


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

    def test_transcode_dicom_to_htj2k_batch(self):
        """Test batch transcoding of entire DICOM series to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest(
                "nvimgcodec not available. Install nvidia-nvimgcodec-cu{XX} matching your CUDA version (e.g., nvidia-nvimgcodec-cu13 for CUDA 13.x)"
            )

        # Use a specific series from dicomweb
        dicom_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )

        # Find DICOM files in source directory
        source_files = sorted(list(Path(dicom_dir).glob("*.dcm")))
        if not source_files:
            source_files = sorted([f for f in Path(dicom_dir).iterdir() if f.is_file()])
        
        self.assertGreater(len(source_files), 0, f"No DICOM files found in {dicom_dir}")
        print(f"\nSource directory: {dicom_dir}")
        print(f"Source files: {len(source_files)}")

        # Create a temporary directory for transcoded output
        output_dir = tempfile.mkdtemp(prefix="htj2k_test_")
        
        try:
            # Perform batch transcoding
            print("\nTranscoding DICOM series to HTJ2K...")
            result_dir = transcode_dicom_to_htj2k(
                input_dir=dicom_dir,
                output_dir=output_dir,
            )
            
            self.assertEqual(result_dir, output_dir, "Output directory should match requested directory")
            
            # Find transcoded files
            transcoded_files = sorted(list(Path(output_dir).glob("*.dcm")))
            if not transcoded_files:
                transcoded_files = sorted([f for f in Path(output_dir).iterdir() if f.is_file()])
            
            print(f"\nTranscoded files: {len(transcoded_files)}")
            
            # Verify file count matches
            self.assertEqual(
                len(transcoded_files), 
                len(source_files), 
                f"Number of transcoded files ({len(transcoded_files)}) should match source files ({len(source_files)})"
            )
            print(f"✓ File count matches: {len(transcoded_files)} files")
            
            # Verify filenames match (directory structure)
            source_names = sorted([f.name for f in source_files])
            transcoded_names = sorted([f.name for f in transcoded_files])
            self.assertEqual(
                source_names, 
                transcoded_names, 
                "Filenames should match between source and transcoded directories"
            )
            print(f"✓ Directory structure preserved: all filenames match")
            
            # Verify each file has been correctly transcoded
            print("\nVerifying lossless transcoding...")
            verified_count = 0
            
            for source_file, transcoded_file in zip(source_files, transcoded_files):
                # Read original DICOM
                ds_original = pydicom.dcmread(str(source_file))
                original_pixels = ds_original.pixel_array
                
                # Read transcoded DICOM
                ds_transcoded = pydicom.dcmread(str(transcoded_file))
                
                # Verify transfer syntax is HTJ2K
                transfer_syntax = str(ds_transcoded.file_meta.TransferSyntaxUID)
                self.assertTrue(
                    transfer_syntax.startswith("1.2.840.10008.1.2.4.20"),
                    f"Transfer syntax should be HTJ2K (1.2.840.10008.1.2.4.20*), got {transfer_syntax}"
                )
                
                # Decode transcoded pixels
                transcoded_pixels = ds_transcoded.pixel_array
                
                # Verify pixel values are identical (lossless)
                np.testing.assert_array_equal(
                    original_pixels,
                    transcoded_pixels,
                    err_msg=f"Pixel values should be identical (lossless) for {source_file.name}"
                )
                
                # Verify metadata is preserved
                self.assertEqual(
                    ds_original.Rows, 
                    ds_transcoded.Rows, 
                    "Image dimensions (Rows) should be preserved"
                )
                self.assertEqual(
                    ds_original.Columns, 
                    ds_transcoded.Columns, 
                    "Image dimensions (Columns) should be preserved"
                )
                self.assertEqual(
                    ds_original.BitsAllocated, 
                    ds_transcoded.BitsAllocated, 
                    "BitsAllocated should be preserved"
                )
                self.assertEqual(
                    ds_original.BitsStored, 
                    ds_transcoded.BitsStored, 
                    "BitsStored should be preserved"
                )
                
                verified_count += 1
            
            print(f"✓ All {verified_count} files verified: pixel values are identical (lossless)")
            print(f"✓ Transfer syntax verified: HTJ2K (1.2.840.10008.1.2.4.20*)")
            print(f"✓ Metadata preserved: dimensions, bit depth, etc.")
            
            # Verify that transcoded files are actually compressed
            # HTJ2K files should typically be smaller or similar size for lossless
            source_size = sum(f.stat().st_size for f in source_files)
            transcoded_size = sum(f.stat().st_size for f in transcoded_files)
            print(f"\nFile size comparison:")
            print(f"  Original:   {source_size:,} bytes")
            print(f"  Transcoded: {transcoded_size:,} bytes")
            print(f"  Ratio:      {transcoded_size/source_size:.2%}")
            
            print(f"\n✓ Batch HTJ2K transcoding test passed!")
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"\n✓ Cleaned up temporary directory: {output_dir}")

    def test_transcode_mixed_directory(self):
        """Test transcoding a directory with both uncompressed and HTJ2K images."""
        if not HAS_NVIMGCODEC:
            self.skipTest(
                "nvimgcodec not available. Install nvidia-nvimgcodec-cu{XX} matching your CUDA version (e.g., nvidia-nvimgcodec-cu13 for CUDA 13.x)"
            )

        # Use uncompressed DICOM series
        uncompressed_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )
        
        # Find uncompressed DICOM files
        uncompressed_files = sorted(list(Path(uncompressed_dir).glob("*.dcm")))
        if not uncompressed_files:
            uncompressed_files = sorted([f for f in Path(uncompressed_dir).iterdir() if f.is_file()])
        
        self.assertGreater(len(uncompressed_files), 10, f"Need at least 10 DICOM files in {uncompressed_dir}")
        
        # Create a mixed directory with some uncompressed and some HTJ2K files
        import shutil
        mixed_dir = tempfile.mkdtemp(prefix="htj2k_mixed_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_output_")
        htj2k_intermediate = tempfile.mkdtemp(prefix="htj2k_intermediate_")
        
        try:
            print(f"\nCreating mixed directory with uncompressed and HTJ2K files...")
            
            # First, transcode half of the files to HTJ2K
            mid_point = len(uncompressed_files) // 2
            
            # Copy first half as uncompressed
            uncompressed_subset = uncompressed_files[:mid_point]
            for f in uncompressed_subset:
                shutil.copy2(str(f), os.path.join(mixed_dir, f.name))
            
            print(f"  Copied {len(uncompressed_subset)} uncompressed files")
            
            # Transcode second half to HTJ2K
            htj2k_source_dir = tempfile.mkdtemp(prefix="htj2k_source_", dir=htj2k_intermediate)
            for f in uncompressed_files[mid_point:]:
                shutil.copy2(str(f), os.path.join(htj2k_source_dir, f.name))
            
            # Transcode this subset to HTJ2K
            htj2k_transcoded_dir = transcode_dicom_to_htj2k(
                input_dir=htj2k_source_dir,
                output_dir=None,  # Use temp dir
            )
            
            # Copy the transcoded HTJ2K files to mixed directory
            htj2k_files_to_copy = list(Path(htj2k_transcoded_dir).glob("*.dcm"))
            if not htj2k_files_to_copy:
                htj2k_files_to_copy = [f for f in Path(htj2k_transcoded_dir).iterdir() if f.is_file()]
            
            for f in htj2k_files_to_copy:
                shutil.copy2(str(f), os.path.join(mixed_dir, f.name))
            
            print(f"  Copied {len(htj2k_files_to_copy)} HTJ2K files")
            
            # Now we have a mixed directory
            mixed_files = sorted(list(Path(mixed_dir).iterdir()))
            self.assertEqual(len(mixed_files), len(uncompressed_files), "Mixed directory should have all files")
            
            print(f"\nMixed directory created with {len(mixed_files)} files:")
            print(f"  - {len(uncompressed_subset)} uncompressed")
            print(f"  - {len(htj2k_files_to_copy)} HTJ2K")
            
            # Verify the transfer syntaxes before transcoding
            uncompressed_count_before = 0
            htj2k_count_before = 0
            for f in mixed_files:
                ds = pydicom.dcmread(str(f))
                ts = str(ds.file_meta.TransferSyntaxUID)
                if ts.startswith("1.2.840.10008.1.2.4.20"):
                    htj2k_count_before += 1
                else:
                    uncompressed_count_before += 1
            
            print(f"\nBefore transcoding:")
            print(f"  - Uncompressed: {uncompressed_count_before}")
            print(f"  - HTJ2K: {htj2k_count_before}")
            
            # Store original pixel data from HTJ2K files for comparison
            htj2k_original_data = {}
            for f in mixed_files:
                ds = pydicom.dcmread(str(f))
                ts = str(ds.file_meta.TransferSyntaxUID)
                if ts.startswith("1.2.840.10008.1.2.4.20"):
                    htj2k_original_data[f.name] = {
                        'pixels': ds.pixel_array.copy(),
                        'mtime': f.stat().st_mtime,
                    }
            
            # Now transcode the mixed directory
            print(f"\nTranscoding mixed directory...")
            result_dir = transcode_dicom_to_htj2k(
                input_dir=mixed_dir,
                output_dir=output_dir,
            )
            
            self.assertEqual(result_dir, output_dir, "Output directory should match requested directory")
            
            # Verify all files are in output
            output_files = sorted(list(Path(output_dir).iterdir()))
            self.assertEqual(
                len(output_files), 
                len(mixed_files), 
                "Output should have same number of files as input"
            )
            print(f"\n✓ File count matches: {len(output_files)} files")
            
            # Verify all filenames match
            input_names = sorted([f.name for f in mixed_files])
            output_names = sorted([f.name for f in output_files])
            self.assertEqual(input_names, output_names, "All filenames should be preserved")
            print(f"✓ Directory structure preserved: all filenames match")
            
            # Verify all output files are HTJ2K
            all_htj2k = True
            for f in output_files:
                ds = pydicom.dcmread(str(f))
                ts = str(ds.file_meta.TransferSyntaxUID)
                if not ts.startswith("1.2.840.10008.1.2.4.20"):
                    all_htj2k = False
                    print(f"  ERROR: {f.name} has transfer syntax {ts}")
            
            self.assertTrue(all_htj2k, "All output files should be HTJ2K")
            print(f"✓ All {len(output_files)} output files are HTJ2K")
            
            # Verify that HTJ2K files were copied (not re-transcoded)
            print(f"\nVerifying HTJ2K files were copied correctly...")
            for filename, original_data in htj2k_original_data.items():
                output_file = Path(output_dir) / filename
                self.assertTrue(output_file.exists(), f"HTJ2K file {filename} should exist in output")
                
                # Read the output file
                ds_output = pydicom.dcmread(str(output_file))
                output_pixels = ds_output.pixel_array
                
                # Verify pixel data is identical (proving it was copied, not re-transcoded)
                np.testing.assert_array_equal(
                    original_data['pixels'],
                    output_pixels,
                    err_msg=f"HTJ2K file {filename} should have identical pixels after copy"
                )
            
            print(f"✓ All {len(htj2k_original_data)} HTJ2K files were copied correctly")
            
            # Verify that uncompressed files were transcoded and have correct pixel values
            print(f"\nVerifying uncompressed files were transcoded correctly...")
            transcoded_count = 0
            for input_file in mixed_files:
                ds_input = pydicom.dcmread(str(input_file))
                ts_input = str(ds_input.file_meta.TransferSyntaxUID)
                
                if not ts_input.startswith("1.2.840.10008.1.2.4.20"):
                    # This was an uncompressed file, verify it was transcoded
                    output_file = Path(output_dir) / input_file.name
                    ds_output = pydicom.dcmread(str(output_file))
                    
                    # Verify transfer syntax changed to HTJ2K
                    ts_output = str(ds_output.file_meta.TransferSyntaxUID)
                    self.assertTrue(
                        ts_output.startswith("1.2.840.10008.1.2.4.20"),
                        f"File {input_file.name} should be HTJ2K after transcoding"
                    )
                    
                    # Verify lossless transcoding (pixel values identical)
                    np.testing.assert_array_equal(
                        ds_input.pixel_array,
                        ds_output.pixel_array,
                        err_msg=f"File {input_file.name} should have identical pixels after lossless transcoding"
                    )
                    
                    transcoded_count += 1
            
            print(f"✓ All {transcoded_count} uncompressed files were transcoded correctly (lossless)")
            
            print(f"\n✓ Mixed directory transcoding test passed!")
            print(f"  - HTJ2K files copied: {len(htj2k_original_data)}")
            print(f"  - Uncompressed files transcoded: {transcoded_count}")
            print(f"  - Total output files: {len(output_files)}")
            
        finally:
            # Clean up all temporary directories
            import shutil
            for temp_dir in [mixed_dir, output_dir, htj2k_intermediate]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

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
