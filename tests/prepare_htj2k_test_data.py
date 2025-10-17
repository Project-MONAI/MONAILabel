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
import shutil
import sys
from pathlib import Path

import numpy as np
import pydicom

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the download/extract functions from setup.py
from monai.apps import download_url, extractall

TEST_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_DATA = os.path.join(TEST_DIR, "data")

# Persistent (singleton style) getter for nvimgcodec Decoder and Encoder
_decoder_instance = None
_encoder_instance = None


def get_nvimgcodec_decoder():
    """
    Return a persistent nvimgcodec.Decoder instance.

    Returns:
        nvimgcodec.Decoder: Persistent decoder instance (singleton).
    """
    global _decoder_instance
    if _decoder_instance is None:
        from nvidia import nvimgcodec

        _decoder_instance = nvimgcodec.Decoder()
    return _decoder_instance


def get_nvimgcodec_encoder():
    """
    Return a persistent nvimgcodec.Encoder instance.

    Returns:
        nvimgcodec.Encoder: Persistent encoder instance (singleton).
    """
    global _encoder_instance
    if _encoder_instance is None:
        from nvidia import nvimgcodec

        _encoder_instance = nvimgcodec.Encoder()
    return _encoder_instance


def transcode_to_htj2k(source_path, dest_path, verify=False):
    """
    Transcode a DICOM file to HTJ2K encoding.

    Args:
        source_path (str or Path): Path to the DICOM (.dcm) file to encode.
        dest_path (str or Path): Output file path.
        verify (bool): If True, decode output for correctness verification.

    Returns:
        str: Path to the output file containing the HTJ2K-encoded DICOM.
    """
    from nvidia import nvimgcodec

    ds = pydicom.dcmread(source_path)

    # Use pydicom's pixel_array to decode the source image
    # This way we make sure we cover all transfer syntaxes.
    source_pixel_array = ds.pixel_array

    # Ensure it's a numpy array (not a memoryview or other type)
    if not isinstance(source_pixel_array, np.ndarray):
        source_pixel_array = np.array(source_pixel_array)

    # Add channel dimension if needed (nvImageCodec expects shape like (H, W, C))
    if source_pixel_array.ndim == 2:
        source_pixel_array = source_pixel_array[:, :, np.newaxis]

    # nvImageCodec expects a list of images
    decoded_images = [source_pixel_array]

    # Encode to htj2k
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params.num_resolutions = 6
    jpeg2k_encode_params.code_block_size = (64, 64)
    jpeg2k_encode_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2
    jpeg2k_encode_params.prog_order = nvimgcodec.Jpeg2kProgOrder.LRCP
    jpeg2k_encode_params.ht = True

    encoded_htj2k_images = get_nvimgcodec_encoder().encode(
        decoded_images,
        codec="jpeg2k",
        params=nvimgcodec.EncodeParams(
            quality_type=nvimgcodec.QualityType.LOSSLESS,
            jpeg2k_encode_params=jpeg2k_encode_params,
        ),
    )

    # Save to file using pydicom
    new_encoded_frames = [bytes(code_stream) for code_stream in encoded_htj2k_images]
    encapsulated_pixel_data = pydicom.encaps.encapsulate(new_encoded_frames)
    ds.PixelData = encapsulated_pixel_data

    # HTJ2K Lossless Only Transfer Syntax UID
    ds.file_meta.TransferSyntaxUID = pydicom.uid.UID("1.2.840.10008.1.2.4.201")

    # Ensure destination directory exists
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(dest_path)

    if verify:
        # Decode htj2k to verify correctness
        ds_verify = pydicom.dcmread(dest_path)
        pixel_data = ds_verify.PixelData
        data_sequence = pydicom.encaps.decode_data_sequence(pixel_data)
        images_verify = get_nvimgcodec_decoder().decode(
            data_sequence,
            params=nvimgcodec.DecodeParams(allow_any_depth=True, color_spec=nvimgcodec.ColorSpec.UNCHANGED),
        )
        assert len(images_verify) == 1
        image = np.array(images_verify[0].cpu()).squeeze()  # Remove extra dimension
        assert (
            image.shape == ds_verify.pixel_array.shape
        ), f"Shape mismatch: {image.shape} vs {ds_verify.pixel_array.shape}"
        assert (
            image.dtype == ds_verify.pixel_array.dtype
        ), f"Dtype mismatch: {image.dtype} vs {ds_verify.pixel_array.dtype}"
        assert np.allclose(image, ds_verify.pixel_array), "Pixel values don't match"

    # Print stats
    source_size = os.path.getsize(source_path)
    target_size = os.path.getsize(dest_path)

    def human_readable_size(size, decimal_places=2):
        for unit in ["bytes", "KB", "MB", "GB", "TB"]:
            if size < 1024.0 or unit == "TB":
                return f"{size:.{decimal_places}f} {unit}"
            size /= 1024.0

    print(f"  Encoded: {Path(source_path).name} -> {Path(dest_path).name}")
    print(f"    Original: {human_readable_size(source_size)} | HTJ2K: {human_readable_size(target_size)}", end="")
    size_diff = target_size - source_size
    if size_diff < 0:
        print(f" | Saved: {abs(size_diff)/source_size*100:.1f}%")
    else:
        print(f" | Larger: {size_diff/source_size*100:.1f}%")

    return dest_path


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
        import numpy as np
        import pydicom
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

    # Find all DICOM files recursively in dicomweb directory
    source_dcm_files = list(source_base_dir.rglob("*.dcm"))
    if not source_dcm_files:
        logger.warning(f"No source DICOM files found in {source_base_dir}, skipping HTJ2K creation")
        return

    logger.info(f"Creating HTJ2K test data from {len(source_dcm_files)} dicomweb DICOM files...")

    n_encoded = 0
    n_failed = 0

    for src_file in source_dcm_files:
        # Preserve the exact directory structure from dicomweb
        rel_path = src_file.relative_to(source_base_dir)
        dest_file = htj2k_base_dir / rel_path

        # Create subdirectory if needed
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if dest_file.exists():
            continue

        try:
            transcode_to_htj2k(str(src_file), str(dest_file), verify=False)
            n_encoded += 1
        except Exception as e:
            logger.warning(f"Failed to encode {src_file.name}: {e}")
            n_failed += 1

    if n_encoded > 0:
        logger.info(f"Created {n_encoded} HTJ2K test files in {htj2k_base_dir}")
    if n_failed > 0:
        logger.warning(f"Failed to create {n_failed} HTJ2K files")


def create_htj2k_dataset():
    """Transcode all DICOM files to HTJ2K encoding."""
    print("\n" + "=" * 80)
    print("Step 2: Creating HTJ2K-encoded versions")
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

    source_base = Path(TEST_DATA)
    dest_base = Path(TEST_DATA) / "dataset" / "dicom_htj2k"

    if not source_base.exists():
        print(f"ERROR: Source DICOM data directory not found at: {source_base}")
        print("Run this script first to download the data.")
        return False

    # Find all DICOM files recursively
    dcm_files = list(source_base.rglob("*.dcm"))
    if not dcm_files:
        print(f"ERROR: No DICOM files found in: {source_base}")
        return False

    print(f"Found {len(dcm_files)} DICOM files to transcode")

    n_encoded = 0
    n_skipped = 0
    n_failed = 0

    for src_file in dcm_files:
        # Preserve directory structure
        rel_path = src_file.relative_to(source_base)
        dest_file = dest_base / rel_path

        # Only encode if target doesn't exist
        if dest_file.exists():
            n_skipped += 1
            continue

        try:
            transcode_to_htj2k(str(src_file), str(dest_file), verify=True)
            n_encoded += 1
        except Exception as e:
            print(f"  ERROR encoding {src_file.name}: {e}")
            n_failed += 1

    print(f"\n{'='*80}")
    print(f"HTJ2K encoding complete!")
    print(f"  Encoded: {n_encoded} files")
    print(f"  Skipped (already exist): {n_skipped} files")
    print(f"  Failed: {n_failed} files")
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
