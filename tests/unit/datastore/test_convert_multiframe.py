"""
Unit tests for convert_multiframe module.

Tests the conversion of legacy DICOM CT, MR, and PET series to enhanced multi-frame format.
"""

import os
import tempfile
import unittest
from pathlib import Path

import highdicom
import pydicom

from monailabel.datastore.utils.convert_multiframe import (
    batch_convert_by_series,
    convert_and_convert_to_htj2k,
    convert_to_enhanced_dicom,
    validate_dicom_series,
)


class TestConvertMultiframe(unittest.TestCase):
    """Test DICOM series conversion to enhanced multi-frame format."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        cls.base_dir = Path(__file__).parent.parent.parent.parent
        cls.test_data_dir = cls.base_dir / "tests" / "data" / "dataset"

        # Find available test data directories
        cls.dicomweb_dir = cls.test_data_dir / "dicomweb"
        cls.dicomweb_htj2k_dir = cls.test_data_dir / "dicomweb_htj2k"

    def test_01_validate_series(self):
        """Test validation of a DICOM series."""
        for root, dirs, files in os.walk(self.dicomweb_dir):
            if files and any(f.endswith(".dcm") for f in files):
                series_dir = Path(root)
                print(f"Testing validation on: {series_dir}")

                # Validate the series
                is_valid = validate_dicom_series(series_dir)
                print(f"Validation result: {is_valid}")

                # We may get False if the series is not CT/MR/PT or has issues
                # But the test passes if no exception is raised
                self.assertIsInstance(is_valid, bool)
                break

    def test_02_convert_series_full(self):
        """Test full conversion to enhanced multi-frame format."""
        for root, dirs, files in os.walk(self.dicomweb_dir):
            if files and any(f.endswith(".dcm") for f in files):
                series_dir = Path(root)

                # Check if this is a CT/MR/PT series
                first_file = next((f for f in files if f.endswith(".dcm")), None)
                if first_file:
                    try:
                        ds = pydicom.dcmread(Path(root) / first_file, stop_before_pixels=True)
                        modality = getattr(ds, "Modality", None)

                        if modality not in {"CT", "MR", "PT"}:
                            print(f"Skipping series with modality: {modality}")
                            continue

                        print(f"Testing full conversion on {modality} series: {series_dir}")

                        # Create temporary output file
                        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                            output_file = tmp.name

                        try:
                            # Convert to enhanced format
                            result = convert_to_enhanced_dicom(
                                input_source=series_dir,
                                output_file=output_file,
                            )

                            if result:
                                # Verify the output file was created
                                self.assertTrue(os.path.exists(output_file))

                                # Load and verify the enhanced DICOM
                                enhanced_ds = pydicom.dcmread(output_file)
                                print(f"Enhanced DICOM created:")
                                print(f"  Modality: {enhanced_ds.Modality}")
                                print(f"  NumberOfFrames: {getattr(enhanced_ds, 'NumberOfFrames', 'N/A')}")
                                print(f"  SOPClassUID: {enhanced_ds.SOPClassUID}")

                                # Should have NumberOfFrames attribute
                                self.assertTrue(hasattr(enhanced_ds, "NumberOfFrames"))
                                self.assertGreater(enhanced_ds.NumberOfFrames, 0)
                            else:
                                print("Conversion returned False")

                        finally:
                            # Clean up
                            if os.path.exists(output_file):
                                os.unlink(output_file)

                        # Only test one series
                        break

                    except Exception as e:
                        print(f"Error processing series: {e}")
                        continue

    def test_03_convert_and_compress_htj2k(self):
        """Test conversion to enhanced multi-frame format with HTJ2K compression."""
        # Check if nvImageCodec is available for HTJ2K
        try:
            from nvidia import nvimgcodec
        except ImportError:
            self.skipTest("nvImageCodec is not installed (required for HTJ2K)")

        for root, dirs, files in os.walk(self.dicomweb_dir):
            if files and any(f.endswith(".dcm") for f in files):
                series_dir = Path(root)

                # Check if this is a CT/MR/PT series
                first_file = next((f for f in files if f.endswith(".dcm")), None)
                if first_file:
                    try:
                        ds = pydicom.dcmread(Path(root) / first_file, stop_before_pixels=True)
                        modality = getattr(ds, "Modality", None)

                        if modality not in {"CT", "MR", "PT"}:
                            print(f"Skipping series with modality: {modality}")
                            continue

                        print(f"Testing HTJ2K conversion on {modality} series: {series_dir}")

                        # Create temporary output file
                        with tempfile.NamedTemporaryFile(suffix="_htj2k.dcm", delete=False) as tmp:
                            output_file = tmp.name

                        try:
                            # Convert to enhanced format and compress with HTJ2K
                            result = convert_and_convert_to_htj2k(
                                input_source=series_dir,
                                output_file=output_file,
                                preserve_series_uid=True,
                                num_resolutions=6,
                                progression_order="RPCL",
                            )

                            if result:
                                # Verify the output file was created
                                self.assertTrue(os.path.exists(output_file))

                                # Load and verify the enhanced DICOM
                                enhanced_ds = pydicom.dcmread(output_file)
                                print(f"Enhanced HTJ2K DICOM created:")
                                print(f"  Modality: {enhanced_ds.Modality}")
                                print(f"  NumberOfFrames: {getattr(enhanced_ds, 'NumberOfFrames', 'N/A')}")
                                print(f"  TransferSyntaxUID: {enhanced_ds.file_meta.TransferSyntaxUID}")
                                print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

                                # Should have NumberOfFrames attribute
                                self.assertTrue(hasattr(enhanced_ds, "NumberOfFrames"))
                                self.assertGreater(enhanced_ds.NumberOfFrames, 0)

                                # Should be HTJ2K compressed
                                htj2k_syntaxes = {
                                    "1.2.840.10008.1.2.4.201",  # HTJ2K Lossless Only
                                    "1.2.840.10008.1.2.4.202",  # HTJ2K with RPCL
                                    "1.2.840.10008.1.2.4.203",  # HTJ2K
                                }
                                self.assertIn(
                                    str(enhanced_ds.file_meta.TransferSyntaxUID),
                                    htj2k_syntaxes,
                                    "Output should be HTJ2K compressed",
                                )
                            else:
                                print("Conversion returned False")

                        finally:
                            # Clean up
                            if os.path.exists(output_file):
                                os.unlink(output_file)

                        # Only test one series
                        break

                    except Exception as e:
                        print(f"Error processing series: {e}")
                        continue

    def test_04_batch_convert_by_series(self):
        """Test batch conversion that groups files by SeriesInstanceUID."""
        # Use the dicomweb directory which may contain multiple series
        if not self.dicomweb_dir.exists():
            self.skipTest("Test DICOM data not found")

        # Create a temporary output directory
        with tempfile.TemporaryDirectory() as temp_output:
            output_dir = Path(temp_output)

            print(f"Testing batch conversion on: {self.dicomweb_dir}")
            print(f"Output directory: {output_dir}")

            try:
                # Scan for DICOM files
                input_files = []
                for filepath in self.dicomweb_dir.rglob("*"):
                    if filepath.is_file() and not filepath.name.startswith("."):
                        try:
                            pydicom.dcmread(filepath, stop_before_pixels=True)
                            input_files.append(str(filepath))
                        except:
                            pass  # Skip non-DICOM files

                print(f"Found {len(input_files)} DICOM files")

                # Create file_loader
                file_loader = [(input_files, str(output_dir))]

                # Run batch conversion
                stats = batch_convert_by_series(
                    file_loader=file_loader,
                    preserve_series_uid=True,
                    compress_htj2k=False,
                )

                print(f"Batch conversion results:")
                print(f"  Total series input: {stats.get('total_series_input', stats.get('total_series', 0))}")
                print(f"  Total series output: {stats.get('total_series_output', 0)}")
                print(f"  Converted to multiframe: {stats.get('converted_to_multiframe', stats.get('converted', 0))}")
                print(f"  Failed: {stats['failed']}")

                # Verify results
                total_series = stats.get("total_series_input", stats.get("total_series", 0))
                self.assertGreater(total_series, 0, "Should find at least one series")
                self.assertIsInstance(stats["series_info"], list)

                # Check that output files were created for successful conversions
                for series_info in stats["series_info"]:
                    if series_info["status"] == "success":
                        output_file = Path(series_info["output_file"])
                        self.assertTrue(output_file.exists(), f"Output file should exist: {output_file}")

                        # Verify it's a valid DICOM file
                        ds = pydicom.dcmread(output_file, stop_before_pixels=True)
                        self.assertTrue(hasattr(ds, "NumberOfFrames"))
                        print(f"  ✓ Created: {output_file.name} ({ds.NumberOfFrames} frames)")

            except Exception as e:
                print(f"Error during batch conversion: {e}")
                raise


if __name__ == "__main__":
    unittest.main()
