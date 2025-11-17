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

import logging
import os
import shutil
import tempfile
from pathlib import Path

from monai.apps import download_url, extractall

TEST_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_DATA = os.path.join(TEST_DIR, "data")

logger = logging.getLogger(__name__)


def run_main():
    downloaded_dataset_file = os.path.join(TEST_DIR, "downloads", "dataset.zip")
    dataset_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/test_dataset.zip"
    if not os.path.exists(downloaded_dataset_file):
        download_url(url=dataset_url, filepath=downloaded_dataset_file)
    if not os.path.exists(os.path.join(TEST_DATA, "dataset")):
        extractall(filepath=downloaded_dataset_file, output_dir=TEST_DATA)

    downloaded_pathology_file = os.path.join(TEST_DIR, "downloads", "JP2K-33003-1.svs")
    pathology_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/JP2K-33003-1.svs"
    if not os.path.exists(downloaded_pathology_file):
        download_url(url=pathology_url, filepath=downloaded_pathology_file)
    if not os.path.exists(os.path.join(TEST_DATA, "pathology")):
        os.makedirs(os.path.join(TEST_DATA, "pathology"))
        shutil.copy(downloaded_pathology_file, os.path.join(TEST_DATA, "pathology"))

    downloaded_endoscopy_file = os.path.join(TEST_DIR, "downloads", "endoscopy_frames.zip")
    endoscopy_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/endoscopy_frames.zip"
    if not os.path.exists(downloaded_endoscopy_file):
        download_url(url=endoscopy_url, filepath=downloaded_endoscopy_file)
    if not os.path.exists(os.path.join(TEST_DATA, "endoscopy")):
        os.makedirs(os.path.join(TEST_DATA, "endoscopy"))
        extractall(filepath=downloaded_endoscopy_file, output_dir=os.path.join(TEST_DATA, "endoscopy"))

    downloaded_detection_file = os.path.join(TEST_DIR, "downloads", "detection_dataset.zip")
    dataset_url = "https://github.com/Project-MONAI/MONAILabel/releases/download/data/detection_dataset.zip"
    if not os.path.exists(downloaded_detection_file):
        download_url(url=dataset_url, filepath=downloaded_detection_file)
    if not os.path.exists(os.path.join(TEST_DATA, "detection")):
        os.makedirs(os.path.join(TEST_DATA, "detection"))
        extractall(filepath=downloaded_detection_file, output_dir=os.path.join(TEST_DATA, "detection"))

    # Create HTJ2K-encoded versions of dicomweb test data if nvimgcodec is available
    try:
        import sys

        sys.path.insert(0, TEST_DIR)
        from monailabel.datastore.utils.convert import (
            convert_single_frame_dicom_series_to_multiframe,
            transcode_dicom_to_htj2k,
        )

        # Create regular HTJ2K files (preserving file structure)
        logger.info("Creating HTJ2K test data (single-frame per file)...")
        source_base_dir = Path(TEST_DATA) / "dataset" / "dicomweb"
        htj2k_base_dir = Path(TEST_DATA) / "dataset" / "dicomweb_htj2k"

        if source_base_dir.exists() and not (htj2k_base_dir.exists() and any(htj2k_base_dir.rglob("*.dcm"))):
            series_dirs = [d for d in source_base_dir.rglob("*") if d.is_dir() and any(d.glob("*.dcm"))]
            for series_dir in series_dirs:
                rel_path = series_dir.relative_to(source_base_dir)
                output_series_dir = htj2k_base_dir / rel_path
                if not (output_series_dir.exists() and any(output_series_dir.glob("*.dcm"))):
                    logger.info(f"  Processing series: {rel_path}")
                    transcode_dicom_to_htj2k(
                        input_dir=str(series_dir),
                        output_dir=str(output_series_dir),
                        num_resolutions=6,
                        code_block_size=(64, 64),
                        add_basic_offset_table=False,
                    )
            logger.info(f"✓ HTJ2K test data created at: {htj2k_base_dir}")
        else:
            logger.info("HTJ2K test data already exists, skipping.")

        # Create multi-frame HTJ2K files (one file per series)
        logger.info("Creating multi-frame HTJ2K test data...")
        htj2k_multiframe_dir = Path(TEST_DATA) / "dataset" / "dicomweb_htj2k_multiframe"

        if source_base_dir.exists() and not (
            htj2k_multiframe_dir.exists() and any(htj2k_multiframe_dir.rglob("*.dcm"))
        ):
            convert_single_frame_dicom_series_to_multiframe(
                input_dir=str(source_base_dir),
                output_dir=str(htj2k_multiframe_dir),
                convert_to_htj2k=True,
                num_resolutions=6,
                code_block_size=(64, 64),
            )
            logger.info(f"✓ Multi-frame HTJ2K test data created at: {htj2k_multiframe_dir}")
        else:
            logger.info("Multi-frame HTJ2K test data already exists, skipping.")

    except ImportError as e:
        if "nvidia" in str(e).lower() or "nvimgcodec" in str(e).lower():
            logger.info("Note: nvidia-nvimgcodec not installed. HTJ2K test data will not be created.")
            logger.info("To enable HTJ2K support, install the package matching your CUDA version:")
            logger.info("  pip install nvidia-nvimgcodec-cu{XX}[all]")
            logger.info("  (Replace {XX} with your CUDA major version, e.g., cu13 for CUDA 13.x)")
            logger.info("Installation guide: https://docs.nvidia.com/cuda/nvimagecodec/installation.html")
        else:
            logger.warning(f"Could not import HTJ2K creation module: {e}")
    except Exception as e:
        logger.warning(f"HTJ2K test data creation failed: {e}")
        logger.info("You can manually run: python tests/prepare_htj2k_test_data.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_main()
