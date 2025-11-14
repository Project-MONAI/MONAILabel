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
from pathlib import Path

import numpy as np
import pydicom
from monai.transforms import LoadImage

from monailabel.datastore.utils.convert import dicom_to_nifti
from monailabel.datastore.utils.convert_htj2k import (
    transcode_dicom_to_htj2k,
    convert_single_frame_dicom_series_to_multiframe,
)

# Check if nvimgcodec is available
try:
    from nvidia import nvimgcodec

    HAS_NVIMGCODEC = True
except ImportError:
    HAS_NVIMGCODEC = False
    nvimgcodec = None

# HTJ2K Transfer Syntax UIDs
HTJ2K_TRANSFER_SYNTAXES = frozenset([
    "1.2.840.10008.1.2.4.201",  # High-Throughput JPEG 2000 Image Compression (Lossless Only)
    "1.2.840.10008.1.2.4.202",  # High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only)
    "1.2.840.10008.1.2.4.203",  # High-Throughput JPEG 2000 Image Compression
])


class TestConvertHTJ2K(unittest.TestCase):
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    dicom_dataset = os.path.join(base_dir, "data", "dataset", "dicomweb", "e7567e0a064f0c334226a0658de23afd")

    def test_transcode_multiframe_jpeg_ybr_to_htj2k(self):
        """Test transcoding multi-frame JPEG with YCbCr color space to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest(
                "nvimgcodec not available. Install nvidia-nvimgcodec-cu{XX} matching your CUDA version (e.g., nvidia-nvimgcodec-cu13 for CUDA 13.x)"
            )
        
        # Use pydicom's built-in YBR color multi-frame JPEG example
        import pydicom.data
        
        try:
            source_file = pydicom.data.get_testdata_file("examples_ybr_color.dcm")
        except Exception as e:
            self.skipTest(f"Could not load pydicom test data: {e}")
        
        print(f"\nSource file: {source_file}")
        
        # Create temporary directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_output_")
        
        try:
            # Copy file to input directory
            import shutil
            test_filename = "multiframe_ybr.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original DICOM
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            original_transfer_syntax = str(ds_original.file_meta.TransferSyntaxUID)
            num_frames = int(ds_original.NumberOfFrames) if hasattr(ds_original, 'NumberOfFrames') else 1
            
            print(f"\nOriginal file:")
            print(f"  Transfer Syntax: {original_transfer_syntax}")
            print(f"  Transfer Syntax Name: {ds_original.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_original.PhotometricInterpretation}")
            print(f"  Number of Frames: {num_frames}")
            print(f"  Dimensions: {ds_original.Rows} x {ds_original.Columns}")
            print(f"  Samples Per Pixel: {ds_original.SamplesPerPixel}")
            print(f"  Pixel shape: {original_pixels.shape}")
            print(f"  File size: {os.path.getsize(source_file):,} bytes")
            
            # Perform transcoding
            print(f"\nTranscoding multi-frame YBR JPEG to HTJ2K...")
            import time
            start_time = time.time()
            
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=output_dir,
            )
            
            elapsed_time = time.time() - start_time
            print(f"Transcoding completed in {elapsed_time:.2f} seconds")
            
            self.assertEqual(result_dir, output_dir, "Output directory should match requested directory")
            
            # Find transcoded file
            transcoded_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(transcoded_file), f"Transcoded file should exist: {transcoded_file}")
            
            # Read transcoded DICOM
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_pixels = ds_transcoded.pixel_array
            transcoded_transfer_syntax = str(ds_transcoded.file_meta.TransferSyntaxUID)
            
            print(f"\nTranscoded file:")
            print(f"  Transfer Syntax: {transcoded_transfer_syntax}")
            print(f"  PhotometricInterpretation: {ds_transcoded.PhotometricInterpretation}")
            print(f"  Pixel shape: {transcoded_pixels.shape}")
            print(f"  File size: {os.path.getsize(transcoded_file):,} bytes")
            
            # Verify transfer syntax is HTJ2K
            self.assertIn(
                transcoded_transfer_syntax,
                HTJ2K_TRANSFER_SYNTAXES,
                f"Transfer syntax should be HTJ2K, got {transcoded_transfer_syntax}"
            )
            print(f"✓ Transfer syntax is HTJ2K: {transcoded_transfer_syntax}")
            
            # Verify PhotometricInterpretation was updated to RGB
            self.assertEqual(
                ds_transcoded.PhotometricInterpretation,
                'RGB',
                "PhotometricInterpretation should be updated to RGB after YCbCr conversion"
            )
            print(f"✓ PhotometricInterpretation updated: {ds_original.PhotometricInterpretation} -> {ds_transcoded.PhotometricInterpretation}")
            
            # Verify shapes match
            self.assertEqual(
                original_pixels.shape,
                transcoded_pixels.shape,
                "Pixel array shapes should match"
            )
            print(f"✓ Shapes match: {original_pixels.shape}")
            
            # Verify pixel values are close (allowing small differences due to color space conversions)
            # Use allclose with tolerance since YCbCr->RGB conversion may have rounding differences
            # between pydicom and nvimgcodec (atol=5 allows for typical conversion differences)
            max_diff = np.abs(original_pixels.astype(np.float32) - transcoded_pixels.astype(np.float32)).max()
            mean_diff = np.abs(original_pixels.astype(np.float32) - transcoded_pixels.astype(np.float32)).mean()
            print(f"  Pixel differences: max={max_diff}, mean={mean_diff:.3f}")
            
            if not np.allclose(original_pixels, transcoded_pixels, atol=5, rtol=0):
                print(f"✗ Pixel values differ beyond tolerance")
                self.fail(f"Pixel values should be close (atol=5), but max diff is {max_diff}")
            
            print(f"✓ Pixel values match within tolerance (atol=5, max_diff={max_diff})")
            
            # Verify metadata is preserved
            self.assertEqual(ds_original.Rows, ds_transcoded.Rows, "Rows should be preserved")
            self.assertEqual(ds_original.Columns, ds_transcoded.Columns, "Columns should be preserved")
            self.assertEqual(ds_original.NumberOfFrames, ds_transcoded.NumberOfFrames, "NumberOfFrames should be preserved")
            print(f"✓ Metadata preserved: {num_frames} frames, {ds_original.Rows}x{ds_original.Columns}")
            
            # Compare file sizes
            size_ratio = os.path.getsize(transcoded_file) / os.path.getsize(source_file)
            print(f"\nCompression ratio: {size_ratio:.2%}")
            
            print(f"\n✓ Multi-frame YBR JPEG to HTJ2K transcoding test passed!")
            
        finally:
            # Clean up temporary directories
            import shutil
            for temp_dir in [input_dir, output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    def test_transcode_ct_example_to_htj2k(self):
        """Test transcoding uncompressed CT grayscale image to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('ct'))
        print(f"\nSource: {source_file}")
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_ct_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_ct_output_")
        
        try:
            test_filename = "ct_small.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            
            print(f"Original: {ds_original.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_original.PhotometricInterpretation}")
            print(f"  Shape: {original_pixels.shape}")
            
            # Transcode
            result_dir = transcode_dicom_to_htj2k(input_dir=input_dir, output_dir=output_dir)
            self.assertEqual(result_dir, output_dir)
            
            # Read transcoded
            transcoded_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(transcoded_file))
            
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_pixels = ds_transcoded.pixel_array
            
            print(f"Transcoded: {ds_transcoded.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_transcoded.PhotometricInterpretation}")
            
            # Verify HTJ2K
            self.assertIn(str(ds_transcoded.file_meta.TransferSyntaxUID), HTJ2K_TRANSFER_SYNTAXES)
            
            # Verify lossless (grayscale should be exact)
            np.testing.assert_array_equal(original_pixels, transcoded_pixels)
            print("✓ CT grayscale lossless transcoding verified")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_transcode_mr_example_to_htj2k(self):
        """Test transcoding uncompressed MR grayscale image to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('mr'))
        print(f"\nSource: {source_file}")
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_mr_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_mr_output_")
        
        try:
            test_filename = "mr_small.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            
            print(f"Original: {ds_original.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_original.PhotometricInterpretation}")
            print(f"  Shape: {original_pixels.shape}")
            
            # Transcode
            result_dir = transcode_dicom_to_htj2k(input_dir=input_dir, output_dir=output_dir)
            self.assertEqual(result_dir, output_dir)
            
            # Read transcoded
            transcoded_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(transcoded_file))
            
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_pixels = ds_transcoded.pixel_array
            
            print(f"Transcoded: {ds_transcoded.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_transcoded.PhotometricInterpretation}")
            
            # Verify HTJ2K
            self.assertIn(str(ds_transcoded.file_meta.TransferSyntaxUID), HTJ2K_TRANSFER_SYNTAXES)
            
            # Verify lossless (grayscale should be exact)
            np.testing.assert_array_equal(original_pixels, transcoded_pixels)
            print("✓ MR grayscale lossless transcoding verified")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_transcode_rgb_color_example_to_htj2k(self):
        """Test transcoding uncompressed RGB color image to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('rgb_color'))
        print(f"\nSource: {source_file}")
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_rgb_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_rgb_output_")
        
        try:
            test_filename = "rgb_color.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            
            print(f"Original: {ds_original.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_original.PhotometricInterpretation}")
            print(f"  Shape: {original_pixels.shape}")
            
            # Transcode
            result_dir = transcode_dicom_to_htj2k(input_dir=input_dir, output_dir=output_dir)
            self.assertEqual(result_dir, output_dir)
            
            # Read transcoded
            transcoded_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(transcoded_file))
            
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_pixels = ds_transcoded.pixel_array
            
            print(f"Transcoded: {ds_transcoded.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_transcoded.PhotometricInterpretation}")
            
            # Verify HTJ2K
            self.assertIn(str(ds_transcoded.file_meta.TransferSyntaxUID), HTJ2K_TRANSFER_SYNTAXES)
            
            # Verify PhotometricInterpretation stays RGB
            self.assertEqual(ds_transcoded.PhotometricInterpretation, 'RGB')
            
            # Verify lossless (RGB uncompressed should be exact)
            np.testing.assert_array_equal(original_pixels, transcoded_pixels)
            print("✓ RGB color lossless transcoding verified")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_transcode_jpeg2k_example_to_htj2k(self):
        """Test transcoding JPEG 2000 (YBR_RCT) color image to HTJ2K."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('jpeg2k'))
        print(f"\nSource: {source_file}")
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_jpeg2k_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_jpeg2k_output_")
        
        try:
            test_filename = "jpeg2k.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            
            print(f"Original: {ds_original.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_original.PhotometricInterpretation}")
            print(f"  Shape: {original_pixels.shape}")
            
            # Transcode
            result_dir = transcode_dicom_to_htj2k(input_dir=input_dir, output_dir=output_dir)
            self.assertEqual(result_dir, output_dir)
            
            # Read transcoded
            transcoded_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(transcoded_file))
            
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_pixels = ds_transcoded.pixel_array
            
            print(f"Transcoded: {ds_transcoded.file_meta.TransferSyntaxUID.name}")
            print(f"  PhotometricInterpretation: {ds_transcoded.PhotometricInterpretation}")
            
            # Verify HTJ2K
            self.assertIn(str(ds_transcoded.file_meta.TransferSyntaxUID), HTJ2K_TRANSFER_SYNTAXES)
            
            # Verify PhotometricInterpretation updated to RGB (from YBR_RCT)
            self.assertEqual(ds_transcoded.PhotometricInterpretation, 'RGB')
            print(f"✓ PhotometricInterpretation updated: {ds_original.PhotometricInterpretation} -> RGB")
            
            # Verify pixels match within tolerance (color space conversion may have small differences)
            max_diff = np.abs(original_pixels.astype(np.float32) - transcoded_pixels.astype(np.float32)).max()
            mean_diff = np.abs(original_pixels.astype(np.float32) - transcoded_pixels.astype(np.float32)).mean()
            print(f"  Pixel differences: max={max_diff}, mean={mean_diff:.3f}")
            
            # YBR_RCT is reversible, so differences should be minimal
            self.assertTrue(np.allclose(original_pixels, transcoded_pixels, atol=5, rtol=0))
            print(f"✓ JPEG2K (YBR_RCT) to HTJ2K transcoding verified (max_diff={max_diff})")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

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
                self.assertIn(
                    transfer_syntax,
                    HTJ2K_TRANSFER_SYNTAXES,
                    f"Transfer syntax should be HTJ2K, got {transfer_syntax}"
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
                if ts in HTJ2K_TRANSFER_SYNTAXES:
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
                if ts in HTJ2K_TRANSFER_SYNTAXES:
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
                if ts not in HTJ2K_TRANSFER_SYNTAXES:
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
                
                if ts_input not in HTJ2K_TRANSFER_SYNTAXES:
                    # This was an uncompressed file, verify it was transcoded
                    output_file = Path(output_dir) / input_file.name
                    ds_output = pydicom.dcmread(str(output_file))
                    
                    # Verify transfer syntax changed to HTJ2K
                    ts_output = str(ds_output.file_meta.TransferSyntaxUID)
                    self.assertIn(
                        ts_output,
                        HTJ2K_TRANSFER_SYNTAXES,
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

    def test_transcode_dicom_to_htj2k_multiframe_metadata(self):
        """Test that multi-frame HTJ2K files preserve correct DICOM metadata from original files."""
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

        # Load original DICOM files and sort by Z-coordinate (same as transcode function does)
        source_files = sorted(list(Path(dicom_dir).glob("*.dcm")))
        if not source_files:
            source_files = sorted([f for f in Path(dicom_dir).iterdir() if f.is_file()])

        print(f"\nLoading {len(source_files)} original DICOM files...")
        original_datasets = []
        for source_file in source_files:
            ds = pydicom.dcmread(str(source_file))
            z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0
            original_datasets.append((z_pos, ds))

        # Sort by Z position (same as convert_single_frame_dicom_series_to_multiframe does)
        original_datasets.sort(key=lambda x: x[0])
        original_datasets = [ds for _, ds in original_datasets]
        print(f"✓ Original files loaded and sorted by Z-coordinate")

        # Create temporary output directory
        output_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_metadata_")

        try:
            # Transcode to multi-frame
            result_dir = convert_single_frame_dicom_series_to_multiframe(
                input_dir=dicom_dir,
                output_dir=output_dir,
                convert_to_htj2k=True,
            )

            # Find the multi-frame file
            multiframe_files = list(Path(output_dir).rglob("*.dcm"))
            self.assertEqual(len(multiframe_files), 1, "Should have one multi-frame file")

            # Load the multi-frame file
            ds_multiframe = pydicom.dcmread(str(multiframe_files[0]))

            print(f"\nVerifying multi-frame metadata against original files...")

            # Check NumberOfFrames matches source file count
            self.assertTrue(hasattr(ds_multiframe, "NumberOfFrames"), "Should have NumberOfFrames")
            num_frames = int(ds_multiframe.NumberOfFrames)
            self.assertEqual(num_frames, len(original_datasets), "NumberOfFrames should match source file count")
            print(f"✓ NumberOfFrames: {num_frames} (matches source)")

            # Check FrameIncrementPointer (required for multi-frame)
            self.assertTrue(hasattr(ds_multiframe, "FrameIncrementPointer"), "Should have FrameIncrementPointer")
            self.assertEqual(ds_multiframe.FrameIncrementPointer, 0x00200032, "Should point to ImagePositionPatient")
            print(f"✓ FrameIncrementPointer: {hex(ds_multiframe.FrameIncrementPointer)} (ImagePositionPatient)")

            # Verify top-level metadata matches first frame
            first_original = original_datasets[0]

            # Check ImagePositionPatient (top-level should match first frame)
            self.assertTrue(hasattr(ds_multiframe, "ImagePositionPatient"), "Should have ImagePositionPatient")
            np.testing.assert_array_almost_equal(
                np.array([float(x) for x in ds_multiframe.ImagePositionPatient]),
                np.array([float(x) for x in first_original.ImagePositionPatient]),
                decimal=6,
                err_msg="Top-level ImagePositionPatient should match first original file"
            )
            print(f"✓ ImagePositionPatient matches first frame: {ds_multiframe.ImagePositionPatient}")

            # Check ImageOrientationPatient
            self.assertTrue(hasattr(ds_multiframe, "ImageOrientationPatient"), "Should have ImageOrientationPatient")
            np.testing.assert_array_almost_equal(
                np.array([float(x) for x in ds_multiframe.ImageOrientationPatient]),
                np.array([float(x) for x in first_original.ImageOrientationPatient]),
                decimal=6,
                err_msg="ImageOrientationPatient should match original"
            )
            print(f"✓ ImageOrientationPatient matches original: {ds_multiframe.ImageOrientationPatient}")

            # Check PixelSpacing
            self.assertTrue(hasattr(ds_multiframe, "PixelSpacing"), "Should have PixelSpacing")
            np.testing.assert_array_almost_equal(
                np.array([float(x) for x in ds_multiframe.PixelSpacing]),
                np.array([float(x) for x in first_original.PixelSpacing]),
                decimal=6,
                err_msg="PixelSpacing should match original"
            )
            print(f"✓ PixelSpacing matches original: {ds_multiframe.PixelSpacing}")

            # Check SliceThickness
            if hasattr(first_original, "SliceThickness"):
                self.assertTrue(hasattr(ds_multiframe, "SliceThickness"), "Should have SliceThickness")
                self.assertAlmostEqual(
                    float(ds_multiframe.SliceThickness),
                    float(first_original.SliceThickness),
                    places=6,
                    msg="SliceThickness should match original"
                )
                print(f"✓ SliceThickness matches original: {ds_multiframe.SliceThickness}")

            # Check for PerFrameFunctionalGroupsSequence
            self.assertTrue(
                hasattr(ds_multiframe, "PerFrameFunctionalGroupsSequence"),
                "Should have PerFrameFunctionalGroupsSequence"
            )
            per_frame_seq = ds_multiframe.PerFrameFunctionalGroupsSequence
            self.assertEqual(
                len(per_frame_seq),
                num_frames,
                f"PerFrameFunctionalGroupsSequence should have {num_frames} items"
            )
            print(f"✓ PerFrameFunctionalGroupsSequence: {len(per_frame_seq)} frames")

            # Verify each frame's metadata matches corresponding original file
            print(f"\nVerifying per-frame metadata...")
            mismatches = []
            for frame_idx in range(num_frames):
                frame_item = per_frame_seq[frame_idx]
                original_ds = original_datasets[frame_idx]

                # Check PlanePositionSequence
                self.assertTrue(
                    hasattr(frame_item, "PlanePositionSequence"),
                    f"Frame {frame_idx} should have PlanePositionSequence"
                )
                plane_pos = frame_item.PlanePositionSequence[0]
                self.assertTrue(
                    hasattr(plane_pos, "ImagePositionPatient"),
                    f"Frame {frame_idx} should have ImagePositionPatient in PlanePositionSequence"
                )

                # Verify ImagePositionPatient matches original
                multiframe_ipp = np.array([float(x) for x in plane_pos.ImagePositionPatient])
                original_ipp = np.array([float(x) for x in original_ds.ImagePositionPatient])
                
                try:
                    np.testing.assert_array_almost_equal(
                        multiframe_ipp,
                        original_ipp,
                        decimal=6,
                        err_msg=f"Frame {frame_idx} ImagePositionPatient should match original"
                    )
                except AssertionError as e:
                    mismatches.append(f"Frame {frame_idx}: {e}")

                # Check PlaneOrientationSequence
                self.assertTrue(
                    hasattr(frame_item, "PlaneOrientationSequence"),
                    f"Frame {frame_idx} should have PlaneOrientationSequence"
                )
                plane_orient = frame_item.PlaneOrientationSequence[0]
                self.assertTrue(
                    hasattr(plane_orient, "ImageOrientationPatient"),
                    f"Frame {frame_idx} should have ImageOrientationPatient in PlaneOrientationSequence"
                )

                # Verify ImageOrientationPatient matches original
                multiframe_iop = np.array([float(x) for x in plane_orient.ImageOrientationPatient])
                original_iop = np.array([float(x) for x in original_ds.ImageOrientationPatient])
                
                try:
                    np.testing.assert_array_almost_equal(
                        multiframe_iop,
                        original_iop,
                        decimal=6,
                        err_msg=f"Frame {frame_idx} ImageOrientationPatient should match original"
                    )
                except AssertionError as e:
                    mismatches.append(f"Frame {frame_idx}: {e}")

            # Report any mismatches
            if mismatches:
                self.fail(f"Per-frame metadata mismatches:\n" + "\n".join(mismatches))

            print(f"✓ All {num_frames} frames have metadata matching original files")

            # Verify frame ordering (first and last frame positions)
            first_frame_pos = per_frame_seq[0].PlanePositionSequence[0].ImagePositionPatient
            last_frame_pos = per_frame_seq[-1].PlanePositionSequence[0].ImagePositionPatient

            first_original_pos = original_datasets[0].ImagePositionPatient
            last_original_pos = original_datasets[-1].ImagePositionPatient

            print(f"\nFrame ordering verification:")
            print(f"  First frame Z: {first_frame_pos[2]} (original: {first_original_pos[2]})")
            print(f"  Last frame Z:  {last_frame_pos[2]} (original: {last_original_pos[2]})")

            # Verify positions match originals
            self.assertAlmostEqual(
                float(first_frame_pos[2]),
                float(first_original_pos[2]),
                places=6,
                msg="First frame Z should match first original"
            )
            self.assertAlmostEqual(
                float(last_frame_pos[2]),
                float(last_original_pos[2]),
                places=6,
                msg="Last frame Z should match last original"
            )
            print(f"✓ Frame ordering matches original files")

            print(f"\n✓ Multi-frame metadata test passed - all metadata preserved correctly!")

        finally:
            # Clean up
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_transcode_dicom_to_htj2k_multiframe_lossless(self):
        """Test that multi-frame HTJ2K transcoding is lossless."""
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

        # Load original files
        source_files = sorted(list(Path(dicom_dir).glob("*.dcm")))
        if not source_files:
            source_files = sorted([f for f in Path(dicom_dir).iterdir() if f.is_file()])

        print(f"\nLoading {len(source_files)} original DICOM files...")

        # Read original pixel data and sort by ImagePositionPatient Z-coordinate
        original_frames = []
        for source_file in source_files:
            ds = pydicom.dcmread(str(source_file))
            z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0
            original_frames.append((z_pos, ds.pixel_array.copy()))

        # Sort by Z position (same as convert_single_frame_dicom_series_to_multiframe does)
        original_frames.sort(key=lambda x: x[0])
        original_pixel_stack = np.stack([frame for _, frame in original_frames], axis=0)

        print(f"✓ Original pixel data loaded: {original_pixel_stack.shape}")

        # Create temporary output directory
        output_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_lossless_")

        try:
            # Transcode to multi-frame HTJ2K
            print(f"\nTranscoding to multi-frame HTJ2K...")
            result_dir = convert_single_frame_dicom_series_to_multiframe(
                input_dir=dicom_dir,
                output_dir=output_dir,
                convert_to_htj2k=True,
            )

            # Find the multi-frame file
            multiframe_files = list(Path(output_dir).rglob("*.dcm"))
            self.assertEqual(len(multiframe_files), 1, "Should have one multi-frame file")

            # Load the multi-frame file
            ds_multiframe = pydicom.dcmread(str(multiframe_files[0]))
            multiframe_pixels = ds_multiframe.pixel_array

            print(f"✓ Multi-frame pixel data loaded: {multiframe_pixels.shape}")

            # Verify shapes match
            self.assertEqual(
                multiframe_pixels.shape,
                original_pixel_stack.shape,
                "Multi-frame shape should match original stacked shape"
            )

            # Verify pixel values are identical (lossless)
            print(f"\nVerifying lossless transcoding...")
            np.testing.assert_array_equal(
                original_pixel_stack,
                multiframe_pixels,
                err_msg="Multi-frame pixel values should be identical to original (lossless)"
            )

            print(f"✓ All {len(source_files)} frames are identical (lossless compression verified)")

            # Verify each frame individually
            for frame_idx in range(len(source_files)):
                np.testing.assert_array_equal(
                    original_pixel_stack[frame_idx],
                    multiframe_pixels[frame_idx],
                    err_msg=f"Frame {frame_idx} should be identical"
                )

            print(f"✓ Individual frame verification passed for all {len(source_files)} frames")

            print(f"\n✓ Lossless multi-frame HTJ2K transcoding test passed!")

        finally:
            # Clean up
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_transcode_dicom_to_htj2k_multiframe_nifti_consistency(self):
        """Test that multi-frame HTJ2K produces same NIfTI output as original series."""
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

        print(f"\nConverting original DICOM series to NIfTI...")
        nifti_from_original = dicom_to_nifti(dicom_dir)

        # Create temporary output directory for multi-frame
        output_dir = tempfile.mkdtemp(prefix="htj2k_multiframe_nifti_")

        try:
            # Transcode to multi-frame HTJ2K
            print(f"\nTranscoding to multi-frame HTJ2K...")
            result_dir = convert_single_frame_dicom_series_to_multiframe(
                input_dir=dicom_dir,
                output_dir=output_dir,
                convert_to_htj2k=True,
            )

            # Find the multi-frame file
            multiframe_files = list(Path(output_dir).rglob("*.dcm"))
            self.assertEqual(len(multiframe_files), 1, "Should have one multi-frame file")
            multiframe_dir = multiframe_files[0].parent

            # Convert multi-frame to NIfTI
            print(f"\nConverting multi-frame HTJ2K to NIfTI...")
            nifti_from_multiframe = dicom_to_nifti(str(multiframe_dir))

            # Load both NIfTI files
            data_original = LoadImage(image_only=True)(nifti_from_original)
            data_multiframe = LoadImage(image_only=True)(nifti_from_multiframe)

            print(f"\nComparing NIfTI outputs...")
            print(f"  Original shape:    {data_original.shape}")
            print(f"  Multi-frame shape: {data_multiframe.shape}")

            # Verify shapes match
            self.assertEqual(
                data_original.shape,
                data_multiframe.shape,
                "Original and multi-frame should produce same NIfTI shape"
            )

            # Verify data types match
            self.assertEqual(
                data_original.dtype,
                data_multiframe.dtype,
                "Original and multi-frame should produce same NIfTI data type"
            )

            # Verify pixel values are identical
            np.testing.assert_array_equal(
                data_original,
                data_multiframe,
                err_msg="Original and multi-frame should produce identical NIfTI pixel values"
            )

            print(f"✓ NIfTI outputs are identical")
            print(f"  Shape: {data_original.shape}")
            print(f"  Data type: {data_original.dtype}")
            print(f"  Pixel values: Identical")

            print(f"\n✓ Multi-frame HTJ2K NIfTI consistency test passed!")

        finally:
            # Clean up
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            if os.path.exists(nifti_from_original):
                os.unlink(nifti_from_original)
            if os.path.exists(nifti_from_multiframe):
                os.unlink(nifti_from_multiframe)


    def test_default_progression_order(self):
        """Test that the default progression order is RPCL."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('ct'))
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_default_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_default_output_")
        
        try:
            test_filename = "ct_small.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Transcode WITHOUT specifying progression_order (should default to RPCL)
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Read transcoded
            transcoded_file = os.path.join(output_dir, test_filename)
            ds_transcoded = pydicom.dcmread(transcoded_file)
            transcoded_ts = str(ds_transcoded.file_meta.TransferSyntaxUID)
            
            # Default should be RPCL which uses transfer syntax 1.2.840.10008.1.2.4.202
            expected_ts = "1.2.840.10008.1.2.4.202"
            self.assertEqual(
                transcoded_ts,
                expected_ts,
                f"Default progression order should produce transfer syntax {expected_ts} (RPCL)"
            )
            print(f"✓ Default progression order is RPCL (transfer syntax: {transcoded_ts})")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_progression_order_options(self):
        """Test that all 5 progression orders work correctly with grayscale images."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('ct'))
        
        # Test all 5 progression orders
        progression_orders = [
            ("LRCP", "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only) - quality scalability
            ("RLCP", "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only) - resolution scalability
            ("RPCL", "1.2.840.10008.1.2.4.202"),  # HTJ2K with RPCL Options - progressive by resolution
            ("PCRL", "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only) - progressive by spatial area
            ("CPRL", "1.2.840.10008.1.2.4.201"),  # HTJ2K (Lossless Only) - component scalability
        ]
        
        for prog_order, expected_ts in progression_orders:
            with self.subTest(progression_order=prog_order):
                print(f"\nTesting progression_order={prog_order}")
                
                # Create temp directories
                input_dir = tempfile.mkdtemp(prefix=f"htj2k_{prog_order.lower()}_input_")
                output_dir = tempfile.mkdtemp(prefix=f"htj2k_{prog_order.lower()}_output_")
                
                try:
                    test_filename = "ct_small.dcm"
                    shutil.copy2(source_file, os.path.join(input_dir, test_filename))
                    
                    # Read original
                    ds_original = pydicom.dcmread(source_file)
                    original_pixels = ds_original.pixel_array.copy()
                    
                    # Transcode with specific progression order
                    result_dir = transcode_dicom_to_htj2k(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        progression_order=prog_order
                    )
                    self.assertEqual(result_dir, output_dir)
                    
                    # Read transcoded
                    transcoded_file = os.path.join(output_dir, test_filename)
                    self.assertTrue(os.path.exists(transcoded_file))
                    
                    ds_transcoded = pydicom.dcmread(transcoded_file)
                    transcoded_pixels = ds_transcoded.pixel_array
                    transcoded_ts = str(ds_transcoded.file_meta.TransferSyntaxUID)
                    
                    print(f"  Transfer Syntax: {transcoded_ts}")
                    print(f"  Expected: {expected_ts}")
                    
                    # Verify correct transfer syntax for progression order
                    self.assertEqual(
                        transcoded_ts,
                        expected_ts,
                        f"Transfer syntax should be {expected_ts} for {prog_order}"
                    )
                    
                    # Verify lossless (grayscale should be exact)
                    np.testing.assert_array_equal(original_pixels, transcoded_pixels)
                    print(f"✓ {prog_order} progression order works correctly")
                    
                finally:
                    shutil.rmtree(input_dir, ignore_errors=True)
                    shutil.rmtree(output_dir, ignore_errors=True)

    def test_progression_order_with_ybr_color(self):
        """Test progression orders work correctly with YBR color space conversion."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.data
        
        try:
            source_file = pydicom.data.get_testdata_file("examples_ybr_color.dcm")
        except Exception as e:
            self.skipTest(f"Could not load pydicom test data: {e}")
        
        # Test a subset of progression orders with color images
        # (testing all 5 would take too long, so we test RPCL, LRCP, and RLCP)
        progression_orders = [
            ("RPCL", "1.2.840.10008.1.2.4.202"),  # Default
            ("LRCP", "1.2.840.10008.1.2.4.201"),  # Quality scalability
            ("RLCP", "1.2.840.10008.1.2.4.201"),  # Resolution scalability
        ]
        
        for prog_order, expected_ts in progression_orders:
            with self.subTest(progression_order=prog_order):
                print(f"\nTesting YBR color with progression_order={prog_order}")
                
                import shutil
                input_dir = tempfile.mkdtemp(prefix=f"htj2k_ybr_{prog_order.lower()}_input_")
                output_dir = tempfile.mkdtemp(prefix=f"htj2k_ybr_{prog_order.lower()}_output_")
                
                try:
                    test_filename = "ybr_color.dcm"
                    shutil.copy2(source_file, os.path.join(input_dir, test_filename))
                    
                    # Read original
                    ds_original = pydicom.dcmread(source_file)
                    original_pixels = ds_original.pixel_array.copy()
                    original_pi = ds_original.PhotometricInterpretation
                    
                    # Transcode with specific progression order
                    result_dir = transcode_dicom_to_htj2k(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        progression_order=prog_order
                    )
                    
                    # Read transcoded
                    transcoded_file = os.path.join(output_dir, test_filename)
                    ds_transcoded = pydicom.dcmread(transcoded_file)
                    transcoded_pixels = ds_transcoded.pixel_array
                    transcoded_ts = str(ds_transcoded.file_meta.TransferSyntaxUID)
                    
                    print(f"  Original PI: {original_pi}")
                    print(f"  Transcoded PI: {ds_transcoded.PhotometricInterpretation}")
                    print(f"  Transfer Syntax: {transcoded_ts}")
                    
                    # Verify transfer syntax matches progression order
                    self.assertEqual(transcoded_ts, expected_ts)
                    
                    # Verify PhotometricInterpretation was updated to RGB (from YBR)
                    self.assertEqual(ds_transcoded.PhotometricInterpretation, 'RGB')
                    
                    # Verify pixels match within tolerance (color conversion)
                    max_diff = np.abs(original_pixels.astype(np.float32) - transcoded_pixels.astype(np.float32)).max()
                    self.assertTrue(
                        np.allclose(original_pixels, transcoded_pixels, atol=5, rtol=0),
                        f"Pixels should match within tolerance, max_diff={max_diff}"
                    )
                    print(f"✓ {prog_order} works with YBR color conversion (max_diff={max_diff})")
                    
                finally:
                    shutil.rmtree(input_dir, ignore_errors=True)
                    shutil.rmtree(output_dir, ignore_errors=True)

    def test_progression_order_with_rgb_color(self):
        """Test progression orders work correctly with RGB color images."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('rgb_color'))
        
        # Test a subset of progression orders with RGB images
        progression_orders = [
            ("RPCL", "1.2.840.10008.1.2.4.202"),
            ("PCRL", "1.2.840.10008.1.2.4.201"),  # Position-Component (good for spatial access)
            ("CPRL", "1.2.840.10008.1.2.4.201"),  # Component-Position (good for component access)
        ]
        
        for prog_order, expected_ts in progression_orders:
            with self.subTest(progression_order=prog_order):
                print(f"\nTesting RGB color with progression_order={prog_order}")
                
                input_dir = tempfile.mkdtemp(prefix=f"htj2k_rgb_{prog_order.lower()}_input_")
                output_dir = tempfile.mkdtemp(prefix=f"htj2k_rgb_{prog_order.lower()}_output_")
                
                try:
                    test_filename = "rgb_color.dcm"
                    shutil.copy2(source_file, os.path.join(input_dir, test_filename))
                    
                    # Read original
                    ds_original = pydicom.dcmread(source_file)
                    original_pixels = ds_original.pixel_array.copy()
                    
                    # Transcode with specific progression order
                    result_dir = transcode_dicom_to_htj2k(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        progression_order=prog_order
                    )
                    
                    # Read transcoded
                    transcoded_file = os.path.join(output_dir, test_filename)
                    ds_transcoded = pydicom.dcmread(transcoded_file)
                    transcoded_pixels = ds_transcoded.pixel_array
                    transcoded_ts = str(ds_transcoded.file_meta.TransferSyntaxUID)
                    
                    # Verify transfer syntax matches progression order
                    self.assertEqual(transcoded_ts, expected_ts)
                    
                    # Verify PhotometricInterpretation stays RGB
                    self.assertEqual(ds_transcoded.PhotometricInterpretation, 'RGB')
                    
                    # Verify lossless (RGB uncompressed should be exact)
                    np.testing.assert_array_equal(original_pixels, transcoded_pixels)
                    print(f"✓ {prog_order} works with RGB color images (lossless)")
                    
                finally:
                    shutil.rmtree(input_dir, ignore_errors=True)
                    shutil.rmtree(output_dir, ignore_errors=True)

    def test_invalid_progression_order(self):
        """Test that invalid progression orders raise ValueError."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import pydicom.examples as examples
        import shutil
        
        source_file = str(examples.get_path('ct'))
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_invalid_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_invalid_output_")
        
        try:
            test_filename = "ct_small.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Test various invalid progression orders (lowercase, mixed case, or completely invalid)
            invalid_orders = ["invalid", "rpcl", "lrcp", "rlcp", "pcrl", "cprl", "ABCD", ""]
            
            for invalid_order in invalid_orders:
                with self.subTest(invalid_progression_order=invalid_order):
                    print(f"\nTesting invalid progression_order={repr(invalid_order)}")
                    
                    with self.assertRaises(ValueError) as context:
                        transcode_dicom_to_htj2k(
                            input_dir=input_dir,
                            output_dir=output_dir,
                            progression_order=invalid_order
                        )
                    
                    # Verify error message is helpful and lists all valid options
                    error_msg = str(context.exception)
                    self.assertIn("progression_order", error_msg.lower())
                    # Check that all valid progression orders are mentioned in the error
                    for valid_order in ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]:
                        self.assertIn(valid_order, error_msg)
                    print(f"✓ Correctly raised ValueError: {error_msg}")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_progression_order_multiframe_conversion(self):
        """Test progression orders work with multiframe conversion."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        # Use a specific series from dicomweb
        dicom_dir = os.path.join(
            self.base_dir,
            "data",
            "dataset",
            "dicomweb",
            "e7567e0a064f0c334226a0658de23afd",
            "1.2.826.0.1.3680043.8.274.1.1.8323329.686405.1629744173.656721",
        )
        
        # Test all 5 progression orders
        progression_orders = [
            ("LRCP", "1.2.840.10008.1.2.4.201"),
            ("RLCP", "1.2.840.10008.1.2.4.201"),
            ("RPCL", "1.2.840.10008.1.2.4.202"),
            ("PCRL", "1.2.840.10008.1.2.4.201"),
            ("CPRL", "1.2.840.10008.1.2.4.201"),
        ]
        
        for prog_order, expected_ts in progression_orders:
            with self.subTest(progression_order=prog_order):
                print(f"\nTesting multiframe conversion with progression_order={prog_order}")
                
                output_dir = tempfile.mkdtemp(prefix=f"htj2k_multiframe_{prog_order.lower()}_")
                
                try:
                    # Convert to multiframe with specific progression order
                    result_dir = convert_single_frame_dicom_series_to_multiframe(
                        input_dir=dicom_dir,
                        output_dir=output_dir,
                        convert_to_htj2k=True,
                        progression_order=prog_order
                    )
                    
                    # Find the multi-frame file
                    multiframe_files = list(Path(output_dir).rglob("*.dcm"))
                    self.assertEqual(len(multiframe_files), 1, "Should have one multi-frame file")
                    
                    # Load and verify
                    ds_multiframe = pydicom.dcmread(str(multiframe_files[0]))
                    transcoded_ts = str(ds_multiframe.file_meta.TransferSyntaxUID)
                    
                    print(f"  Transfer Syntax: {transcoded_ts}")
                    print(f"  Expected: {expected_ts}")
                    
                    # Verify correct transfer syntax
                    self.assertEqual(
                        transcoded_ts,
                        expected_ts,
                        f"Transfer syntax should be {expected_ts} for {prog_order}"
                    )
                    
                    print(f"✓ Multiframe conversion with {prog_order} works correctly")
                    
                finally:
                    import shutil
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)

    def test_skip_transfer_syntaxes_htj2k(self):
        """Test skipping files with HTJ2K transfer syntax (copy instead of transcode) - using default skip list."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import shutil
        import time
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_skip_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_skip_output_")
        intermediate_dir = tempfile.mkdtemp(prefix="htj2k_intermediate_")
        
        try:
            # First, create an HTJ2K file by transcoding a test file
            import pydicom.examples as examples
            source_file = str(examples.get_path('ct'))
            test_filename = "ct_htj2k.dcm"
            
            # Copy to intermediate directory
            shutil.copy2(source_file, os.path.join(intermediate_dir, test_filename))
            
            # Transcode to HTJ2K (disable default skip list to force transcoding)
            print("\nStep 1: Creating HTJ2K file...")
            htj2k_dir = tempfile.mkdtemp(prefix="htj2k_created_")
            transcode_dicom_to_htj2k(
                input_dir=intermediate_dir,
                output_dir=htj2k_dir,
                progression_order="RPCL",
                skip_transfer_syntaxes=None  # Override default to force transcoding
            )
            
            # Copy the HTJ2K file to input directory
            htj2k_file = os.path.join(htj2k_dir, test_filename)
            shutil.copy2(htj2k_file, os.path.join(input_dir, test_filename))
            
            # Read the HTJ2K file to get its transfer syntax and checksum
            ds_htj2k = pydicom.dcmread(htj2k_file)
            htj2k_ts = str(ds_htj2k.file_meta.TransferSyntaxUID)
            print(f"HTJ2K Transfer Syntax: {htj2k_ts}")
            
            # Calculate checksum of original HTJ2K file
            import hashlib
            with open(os.path.join(input_dir, test_filename), 'rb') as f:
                original_checksum = hashlib.md5(f.read()).hexdigest()
            original_size = os.path.getsize(os.path.join(input_dir, test_filename))
            original_mtime = os.path.getmtime(os.path.join(input_dir, test_filename))
            
            print(f"\nStep 2: Testing default skip functionality (HTJ2K should be skipped by default)...")
            print(f"  Original file size: {original_size:,} bytes")
            print(f"  Original checksum: {original_checksum}")
            
            # Sleep briefly to ensure timestamps differ if file is modified
            time.sleep(0.1)
            
            # Now transcode with DEFAULT skip_transfer_syntaxes (should skip HTJ2K by default)
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=output_dir
                # Note: NOT passing skip_transfer_syntaxes, using default which includes HTJ2K
            )
            
            self.assertEqual(result_dir, output_dir)
            
            # Verify output file exists
            output_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(output_file), "Output file should exist")
            
            # Calculate checksum of output file
            with open(output_file, 'rb') as f:
                output_checksum = hashlib.md5(f.read()).hexdigest()
            output_size = os.path.getsize(output_file)
            output_mtime = os.path.getmtime(output_file)
            
            print(f"\nStep 3: Verifying file was copied, not transcoded...")
            print(f"  Output file size: {output_size:,} bytes")
            print(f"  Output checksum: {output_checksum}")
            
            # Verify file is identical (not re-encoded)
            self.assertEqual(
                original_checksum,
                output_checksum,
                "File should be identical (copied, not transcoded)"
            )
            self.assertEqual(
                original_size,
                output_size,
                "File size should be identical"
            )
            
            # Verify transfer syntax is still HTJ2K
            ds_output = pydicom.dcmread(output_file)
            self.assertEqual(
                str(ds_output.file_meta.TransferSyntaxUID),
                htj2k_ts,
                "Transfer syntax should be preserved"
            )
            
            print(f"✓ File was copied without transcoding (HTJ2K skipped by default)")
            print(f"✓ Transfer syntax preserved: {htj2k_ts}")
            print(f"✓ Default behavior correctly skips HTJ2K files")
            
        finally:
            # Clean up
            for temp_dir in [input_dir, output_dir, intermediate_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            if 'htj2k_dir' in locals() and os.path.exists(htj2k_dir):
                shutil.rmtree(htj2k_dir, ignore_errors=True)

    def test_skip_transfer_syntaxes_mixed_batch(self):
        """Test mixed batch with some files to skip and some to transcode - using default skip list."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import shutil
        import hashlib
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_mixed_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_mixed_output_")
        
        try:
            # Create two test files: one uncompressed CT, one uncompressed MR
            import pydicom.examples as examples
            ct_source = str(examples.get_path('ct'))
            mr_source = str(examples.get_path('mr'))
            
            ct_filename = "ct_uncompressed.dcm"
            mr_filename = "mr_uncompressed.dcm"
            
            # Copy both to input
            shutil.copy2(ct_source, os.path.join(input_dir, ct_filename))
            shutil.copy2(mr_source, os.path.join(input_dir, mr_filename))
            
            # First pass: transcode CT to HTJ2K with LRCP, keep MR uncompressed
            print("\nStep 1: Creating HTJ2K file with LRCP progression order...")
            first_pass_dir = tempfile.mkdtemp(prefix="htj2k_first_pass_")
            transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=first_pass_dir,
                progression_order="LRCP",  # This will create 1.2.840.10008.1.2.4.201
                skip_transfer_syntaxes=None  # Override default to force transcoding
            )
            
            # Now use the CT HTJ2K file and MR uncompressed for the mixed test
            input_dir2 = tempfile.mkdtemp(prefix="htj2k_mixed2_input_")
            
            # Copy HTJ2K CT to new input
            htj2k_ct_file = os.path.join(first_pass_dir, ct_filename)
            shutil.copy2(htj2k_ct_file, os.path.join(input_dir2, ct_filename))
            
            # Copy uncompressed MR to new input
            shutil.copy2(mr_source, os.path.join(input_dir2, mr_filename))
            
            # Get checksums before processing
            with open(os.path.join(input_dir2, ct_filename), 'rb') as f:
                ct_original_checksum = hashlib.md5(f.read()).hexdigest()
            
            ds_ct = pydicom.dcmread(os.path.join(input_dir2, ct_filename))
            ct_ts = str(ds_ct.file_meta.TransferSyntaxUID)
            
            ds_mr_orig = pydicom.dcmread(os.path.join(input_dir2, mr_filename))
            mr_ts_orig = str(ds_mr_orig.file_meta.TransferSyntaxUID)
            mr_pixels_orig = ds_mr_orig.pixel_array.copy()
            
            print(f"\nStep 2: Processing mixed batch with DEFAULT skip list...")
            print(f"  CT file: {ct_filename} (Transfer Syntax: {ct_ts}) - SKIP (by default)")
            print(f"  MR file: {mr_filename} (Transfer Syntax: {mr_ts_orig}) - TRANSCODE")
            
            # Transcode with DEFAULT skip list (HTJ2K files will be skipped by default)
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir2,
                output_dir=output_dir,
                # Using default skip list which includes HTJ2K formats
                progression_order="RPCL"  # Will use 1.2.840.10008.1.2.4.202 for transcoded files
            )
            
            self.assertEqual(result_dir, output_dir)
            
            print("\nStep 3: Verifying results...")
            
            # Verify CT file was copied (not transcoded)
            ct_output = os.path.join(output_dir, ct_filename)
            self.assertTrue(os.path.exists(ct_output), "CT output should exist")
            
            with open(ct_output, 'rb') as f:
                ct_output_checksum = hashlib.md5(f.read()).hexdigest()
            
            self.assertEqual(
                ct_original_checksum,
                ct_output_checksum,
                "CT file should be identical (copied, not transcoded)"
            )
            
            ds_ct_output = pydicom.dcmread(ct_output)
            self.assertEqual(
                str(ds_ct_output.file_meta.TransferSyntaxUID),
                ct_ts,
                "CT transfer syntax should be preserved (LRCP)"
            )
            print(f"✓ CT file copied without transcoding (HTJ2K skipped by default: {ct_ts})")
            
            # Verify MR file was transcoded to RPCL HTJ2K
            mr_output = os.path.join(output_dir, mr_filename)
            self.assertTrue(os.path.exists(mr_output), "MR output should exist")
            
            ds_mr_output = pydicom.dcmread(mr_output)
            mr_ts_output = str(ds_mr_output.file_meta.TransferSyntaxUID)
            
            self.assertEqual(
                mr_ts_output,
                "1.2.840.10008.1.2.4.202",
                "MR should be transcoded to RPCL HTJ2K"
            )
            
            # Verify MR pixels are lossless
            mr_pixels_output = ds_mr_output.pixel_array
            np.testing.assert_array_equal(
                mr_pixels_orig,
                mr_pixels_output,
                err_msg="MR pixels should be lossless"
            )
            print(f"✓ MR file transcoded to HTJ2K RPCL ({mr_ts_output})")
            print(f"✓ MR pixels are lossless")
            print(f"✓ Mixed batch correctly handles default skip list")
            
        finally:
            # Clean up
            for temp_dir in [input_dir, output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            if 'first_pass_dir' in locals() and os.path.exists(first_pass_dir):
                shutil.rmtree(first_pass_dir, ignore_errors=True)
            if 'input_dir2' in locals() and os.path.exists(input_dir2):
                shutil.rmtree(input_dir2, ignore_errors=True)

    def test_skip_transfer_syntaxes_multiple(self):
        """Test skipping multiple transfer syntax UIDs - using default skip list."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import shutil
        import hashlib
        
        # Create temp directories
        input_dir = tempfile.mkdtemp(prefix="htj2k_multi_skip_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_multi_skip_output_")
        
        try:
            import pydicom.examples as examples
            ct_source = str(examples.get_path('ct'))
            
            # Create two HTJ2K files with different progression orders
            file1_name = "file_lrcp.dcm"
            file2_name = "file_rpcl.dcm"
            
            # Create LRCP HTJ2K file
            print("\nStep 1: Creating LRCP HTJ2K file...")
            temp_dir1 = tempfile.mkdtemp(prefix="htj2k_temp1_")
            shutil.copy2(ct_source, os.path.join(temp_dir1, file1_name))
            htj2k_dir1 = tempfile.mkdtemp(prefix="htj2k_lrcp_")
            transcode_dicom_to_htj2k(
                input_dir=temp_dir1,
                output_dir=htj2k_dir1,
                progression_order="LRCP",
                skip_transfer_syntaxes=None  # Override default to force transcoding
            )
            shutil.copy2(
                os.path.join(htj2k_dir1, file1_name),
                os.path.join(input_dir, file1_name)
            )
            
            # Create RPCL HTJ2K file
            print("Step 2: Creating RPCL HTJ2K file...")
            temp_dir2 = tempfile.mkdtemp(prefix="htj2k_temp2_")
            shutil.copy2(ct_source, os.path.join(temp_dir2, file2_name))
            htj2k_dir2 = tempfile.mkdtemp(prefix="htj2k_rpcl_")
            transcode_dicom_to_htj2k(
                input_dir=temp_dir2,
                output_dir=htj2k_dir2,
                progression_order="RPCL",
                skip_transfer_syntaxes=None  # Override default to force transcoding
            )
            shutil.copy2(
                os.path.join(htj2k_dir2, file2_name),
                os.path.join(input_dir, file2_name)
            )
            
            # Get checksums
            with open(os.path.join(input_dir, file1_name), 'rb') as f:
                checksum1 = hashlib.md5(f.read()).hexdigest()
            with open(os.path.join(input_dir, file2_name), 'rb') as f:
                checksum2 = hashlib.md5(f.read()).hexdigest()
            
            ds1 = pydicom.dcmread(os.path.join(input_dir, file1_name))
            ds2 = pydicom.dcmread(os.path.join(input_dir, file2_name))
            
            ts1 = str(ds1.file_meta.TransferSyntaxUID)
            ts2 = str(ds2.file_meta.TransferSyntaxUID)
            
            print(f"\nStep 3: Processing with DEFAULT skip list (both HTJ2K formats should be skipped)...")
            print(f"  File 1: {file1_name} - {ts1}")
            print(f"  File 2: {file2_name} - {ts2}")
            
            # Use DEFAULT skip list (includes all HTJ2K transfer syntaxes)
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=output_dir
                # Using default skip list which includes all HTJ2K formats
            )
            
            self.assertEqual(result_dir, output_dir)
            
            print("\nStep 4: Verifying both files were copied...")
            
            # Verify both files were copied
            output1 = os.path.join(output_dir, file1_name)
            output2 = os.path.join(output_dir, file2_name)
            
            self.assertTrue(os.path.exists(output1), "File 1 should exist")
            self.assertTrue(os.path.exists(output2), "File 2 should exist")
            
            # Verify checksums match (files were copied, not transcoded)
            with open(output1, 'rb') as f:
                output_checksum1 = hashlib.md5(f.read()).hexdigest()
            with open(output2, 'rb') as f:
                output_checksum2 = hashlib.md5(f.read()).hexdigest()
            
            self.assertEqual(checksum1, output_checksum1, "File 1 should be identical")
            self.assertEqual(checksum2, output_checksum2, "File 2 should be identical")
            
            # Verify transfer syntaxes preserved
            ds_out1 = pydicom.dcmread(output1)
            ds_out2 = pydicom.dcmread(output2)
            
            self.assertEqual(str(ds_out1.file_meta.TransferSyntaxUID), ts1)
            self.assertEqual(str(ds_out2.file_meta.TransferSyntaxUID), ts2)
            
            print(f"✓ Both files copied without transcoding (HTJ2K skipped by default)")
            print(f"✓ File 1 preserved: {ts1}")
            print(f"✓ File 2 preserved: {ts2}")
            print(f"✓ Default skip list correctly handles multiple HTJ2K formats")
            
        finally:
            # Clean up
            for temp_dir in [input_dir, output_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            for var in ['temp_dir1', 'temp_dir2', 'htj2k_dir1', 'htj2k_dir2']:
                if var in locals() and os.path.exists(locals()[var]):
                    shutil.rmtree(locals()[var], ignore_errors=True)

    def test_skip_transfer_syntaxes_override_to_transcode_all(self):
        """Test that explicitly overriding skip list to None/empty transcodes all files (overrides default)."""
        if not HAS_NVIMGCODEC:
            self.skipTest("nvimgcodec not available")
        
        import shutil
        import pydicom.examples as examples
        
        input_dir = tempfile.mkdtemp(prefix="htj2k_override_input_")
        output_dir = tempfile.mkdtemp(prefix="htj2k_override_output_")
        
        try:
            source_file = str(examples.get_path('ct'))
            test_filename = "ct_test.dcm"
            shutil.copy2(source_file, os.path.join(input_dir, test_filename))
            
            # Read original
            ds_original = pydicom.dcmread(source_file)
            original_pixels = ds_original.pixel_array.copy()
            original_ts = str(ds_original.file_meta.TransferSyntaxUID)
            
            print(f"\nOriginal transfer syntax: {original_ts}")
            print(f"Testing override of default skip list to force transcoding...")
            
            # Transcode with None (override default skip list to force transcoding)
            result_dir = transcode_dicom_to_htj2k(
                input_dir=input_dir,
                output_dir=output_dir,
                skip_transfer_syntaxes=None  # Override default to force transcoding
            )
            
            self.assertEqual(result_dir, output_dir)
            
            # Verify file was transcoded
            output_file = os.path.join(output_dir, test_filename)
            self.assertTrue(os.path.exists(output_file))
            
            ds_output = pydicom.dcmread(output_file)
            output_ts = str(ds_output.file_meta.TransferSyntaxUID)
            output_pixels = ds_output.pixel_array
            
            print(f"Output transfer syntax: {output_ts}")
            
            # Should be transcoded to HTJ2K
            self.assertIn(output_ts, HTJ2K_TRANSFER_SYNTAXES)
            self.assertNotEqual(original_ts, output_ts, "Transfer syntax should have changed")
            
            # Pixels should still be lossless
            np.testing.assert_array_equal(original_pixels, output_pixels)
            
            print("✓ Override with None successfully forces transcoding (bypasses default skip list)")
            
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

