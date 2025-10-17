#!/usr/bin/env python3
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

"""
Script to prepare HTJ2K-encoded test data from the dicomweb DICOM dataset.

This script creates HTJ2K-encoded versions of all DICOM files in the
tests/data/dataset/dicomweb/ directory and saves them to a parallel
tests/data/dataset/dicomweb_htj2k/ structure.

The HTJ2K files preserve the exact directory structure:
  dicomweb/<study_id>/<series_uid>/*.dcm
  → dicomweb_htj2k/<study_id>/<series_uid>/*.dcm

This script can be run:
1. Automatically via setup.py (calls create_htj2k_data())
2. Manually: python tests/prepare_htj2k_test_data.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the download/extract functions from setup.py
from monai.apps import download_url, extractall

# Import the transcode function from monailabel
from monailabel.datastore.utils.convert import transcode_dicom_to_htj2k

TEST_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_DATA = os.path.join(TEST_DIR, "data")


def download_and_extract_dicom_data():
    """Download and extract the DICOM test data if not already present."""
    print("=" * 80)
    print("Step 1: Downloading and extracting DICOM test data")
    print("=" * 80)

    downloaded_dicom_file = os.path.join(TEST_DIR, "downloads", "dicom.zip")
    dicom_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/dicom.zip"

    # Download if needed
    if not os.path.exists(downloaded_dicom_file):
        print(f"Downloading: {dicom_url}")
        download_url(url=dicom_url, filepath=downloaded_dicom_file)
        print(f"✓ Downloaded to: {downloaded_dicom_file}")
    else:
        print(f"✓ Already downloaded: {downloaded_dicom_file}")

    # Extract if needed - the zip extracts directly to TEST_DATA
    if not os.path.exists(TEST_DATA) or not any(Path(TEST_DATA).glob("*.dcm")):
        print(f"Extracting to: {TEST_DATA}")
        os.makedirs(TEST_DATA, exist_ok=True)
        extractall(filepath=downloaded_dicom_file, output_dir=TEST_DATA)
        print(f"✓ Extracted DICOM test data")
    else:
        print(f"✓ Already extracted to: {TEST_DATA}")

    return TEST_DATA


def create_htj2k_data(test_data_dir):
    """
    Create HTJ2K-encoded versions of dicomweb test data if not already present.

    This function checks if nvimgcodec is available and creates HTJ2K-encoded
    versions of the dicomweb DICOM files for testing NvDicomReader with HTJ2K compression.
    The HTJ2K files are placed in a parallel dicomweb_htj2k directory structure.
    
    Uses the batch transcoding function from monailabel.datastore.utils.convert for
    improved performance.

    Args:
        test_data_dir: Path to the tests/data directory
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    source_base_dir = Path(test_data_dir) / "dataset" / "dicomweb"
    htj2k_base_dir = Path(test_data_dir) / "dataset" / "dicomweb_htj2k"

    # Check if HTJ2K data already exists
    if htj2k_base_dir.exists() and any(htj2k_base_dir.rglob("*.dcm")):
        logger.info("HTJ2K test data already exists, skipping creation")
        return

    # Check if nvimgcodec is available
    try:
        from nvidia import nvimgcodec
    except ImportError as e:
        logger.info("Note: nvidia-nvimgcodec not installed. HTJ2K test data will not be created.")
        logger.info("To enable HTJ2K support, install the package matching your CUDA version:")
        logger.info("  pip install nvidia-nvimgcodec-cu{XX}[all]")
        logger.info("  (Replace {XX} with your CUDA major version, e.g., cu13 for CUDA 13.x)")
        logger.info("Installation guide: https://docs.nvidia.com/cuda/nvimagecodec/installation.html")
        return

    # Check if source DICOM files exist
    if not source_base_dir.exists():
        logger.warning(f"Source DICOM directory not found: {source_base_dir}")
        return

    logger.info(f"Creating HTJ2K test data from dicomweb DICOM files...")
    logger.info(f"Source: {source_base_dir}")
    logger.info(f"Destination: {htj2k_base_dir}")

    # Process each series directory separately to preserve structure
    series_dirs = [d for d in source_base_dir.rglob("*") if d.is_dir() and any(d.glob("*.dcm"))]
    
    if not series_dirs:
        logger.warning(f"No DICOM series directories found in {source_base_dir}")
        return
    
    logger.info(f"Found {len(series_dirs)} DICOM series directories to process")
    
    total_transcoded = 0
    total_failed = 0
    
    for series_dir in series_dirs:
        try:
            # Calculate relative path and output directory
            rel_path = series_dir.relative_to(source_base_dir)
            output_series_dir = htj2k_base_dir / rel_path
            
            # Skip if already processed
            if output_series_dir.exists() and any(output_series_dir.glob("*.dcm")):
                logger.debug(f"Skipping already processed: {rel_path}")
                continue
            
            logger.info(f"Processing series: {rel_path}")
            
            # Use batch transcoding function
            transcode_dicom_to_htj2k(
                input_dir=str(series_dir),
                output_dir=str(output_series_dir),
                num_resolutions=6,
                code_block_size=(64, 64),
                verify=False,
            )
            
            # Count transcoded files
            transcoded_count = len(list(output_series_dir.glob("*.dcm")))
            total_transcoded += transcoded_count
            logger.info(f"  ✓ Transcoded {transcoded_count} files")
            
        except Exception as e:
            logger.warning(f"Failed to process {series_dir.name}: {e}")
            total_failed += 1

    logger.info(f"\nHTJ2K test data creation complete:")
    logger.info(f"  Successfully processed: {len(series_dirs) - total_failed} series")
    logger.info(f"  Total files transcoded: {total_transcoded}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Output directory: {htj2k_base_dir}")


def create_htj2k_dataset():
    """
    Transcode all DICOM files to HTJ2K encoding.
    
    This is an alternative function for batch transcoding entire datasets.
    For the main test data creation, use create_htj2k_data() instead.
    """
    print("\n" + "=" * 80)
    print("Step 2: Creating HTJ2K-encoded versions (full dataset)")
    print("=" * 80)

    # Check if nvimgcodec is available
    try:
        from nvidia import nvimgcodec

        print("✓ nvImageCodec is available")
    except ImportError:
        print("\n" + "=" * 80)
        print("ERROR: nvImageCodec is not installed")
        print("=" * 80)
        print("\nHTJ2K DICOM encoding requires nvidia-nvimgcodec.")
        print("\nInstall the package matching your CUDA version:")
        print("  pip install nvidia-nvimgcodec-cu{XX}[all]")
        print("\nReplace {XX} with your CUDA major version (e.g., cu13 for CUDA 13.x)")
        print("\nFor installation instructions, visit:")
        print("  https://docs.nvidia.com/cuda/nvimagecodec/installation.html")
        print("=" * 80 + "\n")
        return False

    source_base = Path(TEST_DATA) / "dataset" / "dicomweb"
    dest_base = Path(TEST_DATA) / "dataset" / "dicom_htj2k"

    if not source_base.exists():
        print(f"ERROR: Source DICOM data directory not found at: {source_base}")
        print("Run this script first to download the data.")
        return False

    # Find all series directories with DICOM files
    series_dirs = [d for d in source_base.rglob("*") if d.is_dir() and any(d.glob("*.dcm"))]
    
    if not series_dirs:
        print(f"ERROR: No DICOM series found in: {source_base}")
        return False

    print(f"Found {len(series_dirs)} DICOM series to transcode")

    n_series_encoded = 0
    n_series_skipped = 0
    n_series_failed = 0
    total_files = 0

    for series_dir in series_dirs:
        try:
            # Calculate relative path and output directory
            rel_path = series_dir.relative_to(source_base)
            output_series_dir = dest_base / rel_path
            
            # Skip if already processed
            if output_series_dir.exists() and any(output_series_dir.glob("*.dcm")):
                n_series_skipped += 1
                continue
            
            print(f"\nProcessing series: {rel_path}")
            
            # Use batch transcoding function with verification
            transcode_dicom_to_htj2k(
                input_dir=str(series_dir),
                output_dir=str(output_series_dir),
                num_resolutions=6,
                code_block_size=(64, 64),
                verify=True,  # Enable verification for this function
            )
            
            # Count transcoded files
            file_count = len(list(output_series_dir.glob("*.dcm")))
            total_files += file_count
            n_series_encoded += 1
            print(f"  ✓ Success: {file_count} files")
            
        except Exception as e:
            print(f"  ✗ ERROR processing {series_dir.name}: {e}")
            n_series_failed += 1

    print(f"\n{'='*80}")
    print(f"HTJ2K encoding complete!")
    print(f"  Series encoded: {n_series_encoded}")
    print(f"  Series skipped (already exist): {n_series_skipped}")
    print(f"  Series failed: {n_series_failed}")
    print(f"  Total files transcoded: {total_files}")
    print(f"  Output directory: {dest_base}")
    print(f"{'='*80}")

    # Display directory structure
    if dest_base.exists():
        print("\nHTJ2K-encoded data structure:")
        display_tree(dest_base, max_depth=3)

    return True


def display_tree(directory, prefix="", max_depth=3, current_depth=0):
    """
    Display directory tree structure.

    Args:
        directory (str or Path): Directory to display.
        prefix (str): Tree prefix (for recursion).
        max_depth (int): Max depth to display.
        current_depth (int): Internal use for recursion depth.
    """
    if current_depth >= max_depth:
        return

    try:
        paths = sorted(Path(directory).iterdir(), key=lambda p: (not p.is_dir(), p.name))
        for i, path in enumerate(paths):
            is_last = i == len(paths) - 1
            current_prefix = "└── " if is_last else "├── "

            # Show file count for directories
            if path.is_dir():
                dcm_count = len(list(path.glob("*.dcm")))
                suffix = f" ({dcm_count} .dcm files)" if dcm_count > 0 else ""
                print(f"{prefix}{current_prefix}{path.name}{suffix}")
            else:
                print(f"{prefix}{current_prefix}{path.name}")

            if path.is_dir():
                extension = "    " if is_last else "│   "
                display_tree(path, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass


def main():
    """Main execution function."""
    print("MONAI Label HTJ2K Test Data Preparation")
    print("=" * 80)

    # Create HTJ2K-encoded versions of dicomweb data
    print("\nCreating HTJ2K-encoded versions of dicomweb test data...")
    print("Source: tests/data/dataset/dicomweb/")
    print("Destination: tests/data/dataset/dicomweb_htj2k/")
    print()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    create_htj2k_data(TEST_DATA)

    htj2k_dir = Path(TEST_DATA) / "dataset" / "dicomweb_htj2k"
    if htj2k_dir.exists() and any(htj2k_dir.rglob("*.dcm")):
        print("\n✓ All done! HTJ2K test data is ready.")
        print(f"\nYou can now use the HTJ2K-encoded data from:")
        print(f"  {htj2k_dir}")
        return 0
    else:
        print("\n✗ Failed to create HTJ2K test data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
