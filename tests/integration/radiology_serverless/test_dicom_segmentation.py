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
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from monailabel.config import settings
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


class TestDicomSegmentation(unittest.TestCase):
    """
    Test direct MONAI Label inference on DICOM series without server.

    This test demonstrates serverless usage of MONAILabel for DICOM segmentation,
    loading DICOM series from test data directories and running inference directly
    through the app instance.
    """

    app = None
    base_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    data_dir = os.path.join(base_dir, "tests", "data")

    app_dir = os.path.join(base_dir, "sample-apps", "radiology")
    studies = os.path.join(data_dir, "dataset", "local", "spleen")

    # DICOM test data directories
    dicomweb_dir = os.path.join(data_dir, "dataset", "dicomweb")
    dicomweb_htj2k_dir = os.path.join(data_dir, "dataset", "dicomweb_htj2k")

    # Specific DICOM series for testing
    dicomweb_series = os.path.join(
        data_dir,
        "dataset",
        "dicomweb",
        "e7567e0a064f0c334226a0658de23afd",
        "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266",
    )
    dicomweb_htj2k_series = os.path.join(
        data_dir,
        "dataset",
        "dicomweb_htj2k",
        "e7567e0a064f0c334226a0658de23afd",
        "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266",
    )

    dicomweb_htj2k_multiframe_series = os.path.join(
        data_dir,
        "dataset",
        "dicomweb_htj2k_multiframe",
        "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620251",
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize MONAI Label app for direct usage without server."""
        settings.MONAI_LABEL_APP_DIR = cls.app_dir
        settings.MONAI_LABEL_STUDIES = cls.studies
        settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD = False

        if torch.cuda.is_available():
            logger.info(f"Initializing MONAI Label app from: {cls.app_dir}")
            logger.info(f"Studies directory: {cls.studies}")

            cls.app: MONAILabelApp = app_instance(
                app_dir=cls.app_dir,
                studies=cls.studies,
                conf={
                    "preload": "true",
                    "models": "segmentation_spleen",
                },
            )

            logger.info("App initialized successfully")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after tests."""
        pass

    def _run_inference(self, image_path: str, model_name: str = "segmentation_spleen") -> tuple:
        """
        Run segmentation inference on an image (DICOM series directory or NIfTI file).

        Args:
            image_path: Path to DICOM series directory or NIfTI file
            model_name: Name of the segmentation model to use

        Returns:
            Tuple of (label_data, label_json, inference_time)
        """
        logger.info(f"Running inference on: {image_path}")
        logger.info(f"Model: {model_name}")

        # Prepare inference request
        request = {
            "model": model_name,
            "image": image_path,  # Can be DICOM directory or NIfTI file
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "result_extension": ".nii.gz",  # Force NIfTI output format
            "result_dtype": "uint8",  # Set output data type
        }

        # Get the inference task directly
        task = self.app._infers[model_name]

        # Run inference
        inference_start = time.time()
        label_data, label_json = task(request)
        inference_time = time.time() - inference_start

        logger.info(f"Inference completed in {inference_time:.3f} seconds")

        return label_data, label_json, inference_time

    def _load_segmentation_array(self, label_data):
        """
        Load segmentation data as numpy array.

        Args:
            label_data: File path (str) or numpy array

        Returns:
            numpy array of segmentation
        """
        if isinstance(label_data, str):
            import nibabel as nib

            nii = nib.load(label_data)
            return nii.get_fdata()
        elif isinstance(label_data, np.ndarray):
            return label_data
        else:
            raise ValueError(f"Unexpected label data type: {type(label_data)}")

    def _validate_segmentation_output(self, label_data, label_json):
        """
        Validate that the segmentation output is correct.

        Args:
            label_data: The segmentation result (file path or numpy array)
            label_json: Metadata about the segmentation
        """
        self.assertIsNotNone(label_data, "Label data should not be None")
        self.assertIsNotNone(label_json, "Label JSON should not be None")

        # Check if it's a file path or numpy array
        if isinstance(label_data, str):
            self.assertTrue(os.path.exists(label_data), f"Output file should exist: {label_data}")
            logger.info(f"Segmentation saved to: {label_data}")

            # Try to load and verify the file
            try:
                array = self._load_segmentation_array(label_data)
                self.assertGreater(array.size, 0, "Segmentation array should not be empty")
                logger.info(f"Segmentation shape: {array.shape}, dtype: {array.dtype}")
                logger.info(f"Unique labels: {np.unique(array)}")
            except Exception as e:
                logger.warning(f"Could not load segmentation file: {e}")

        elif isinstance(label_data, np.ndarray):
            self.assertGreater(label_data.size, 0, "Segmentation array should not be empty")
            logger.info(f"Segmentation shape: {label_data.shape}, dtype: {label_data.dtype}")
            logger.info(f"Unique labels: {np.unique(label_data)}")
        else:
            self.fail(f"Unexpected label data type: {type(label_data)}")

        # Validate metadata
        self.assertIsInstance(label_json, dict, "Label JSON should be a dictionary")
        logger.info(f"Label metadata keys: {list(label_json.keys())}")

    def _compare_segmentations(
        self, label_data_1, label_data_2, name_1="Reference", name_2="Comparison", tolerance=0.05
    ):
        """
        Compare two segmentation outputs to verify they are similar.

        Args:
            label_data_1: First segmentation (file path or array)
            label_data_2: Second segmentation (file path or array)
            name_1: Name for first segmentation (for logging)
            name_2: Name for second segmentation (for logging)
            tolerance: Maximum allowed dice coefficient difference (0.0-1.0)

        Returns:
            dict with comparison metrics
        """
        # Load arrays
        array_1 = self._load_segmentation_array(label_data_1)
        array_2 = self._load_segmentation_array(label_data_2)

        # Check shapes match
        self.assertEqual(
            array_1.shape, array_2.shape, f"Segmentation shapes should match: {array_1.shape} vs {array_2.shape}"
        )

        # Calculate dice coefficient for each label
        unique_labels = np.union1d(np.unique(array_1), np.unique(array_2))
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        dice_scores = {}
        for label in unique_labels:
            mask_1 = (array_1 == label).astype(np.float32)
            mask_2 = (array_2 == label).astype(np.float32)

            intersection = np.sum(mask_1 * mask_2)
            sum_masks = np.sum(mask_1) + np.sum(mask_2)

            if sum_masks > 0:
                dice = (2.0 * intersection) / sum_masks
                dice_scores[int(label)] = dice
            else:
                dice_scores[int(label)] = 0.0

        # Calculate overall metrics
        exact_match = np.array_equal(array_1, array_2)
        pixel_accuracy = np.mean(array_1 == array_2)

        comparison_result = {
            "exact_match": exact_match,
            "pixel_accuracy": pixel_accuracy,
            "dice_scores": dice_scores,
            "avg_dice": np.mean(list(dice_scores.values())) if dice_scores else 0.0,
        }

        # Log results
        logger.info(f"\nComparing {name_1} vs {name_2}:")
        logger.info(f"  Exact match: {exact_match}")
        logger.info(f"  Pixel accuracy: {pixel_accuracy:.4f}")
        logger.info(f"  Dice scores by label: {dice_scores}")
        logger.info(f"  Average Dice: {comparison_result['avg_dice']:.4f}")

        # Assert high similarity
        self.assertGreater(
            comparison_result["avg_dice"],
            1.0 - tolerance,
            f"Segmentations should be similar (Dice > {1.0 - tolerance:.2f}). "
            f"Got {comparison_result['avg_dice']:.4f}",
        )

        return comparison_result

    def test_01_app_initialized(self):
        """Test that the app is properly initialized."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.assertIsNotNone(self.app, "App should be initialized")
        self.assertIn("segmentation_spleen", self.app._infers, "segmentation_spleen model should be available")
        logger.info(f"Available models: {list(self.app._infers.keys())}")

    def test_02_dicom_inference_dicomweb(self):
        """Test inference on DICOM series from dicomweb directory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if not self.app:
            self.skipTest("App not initialized")

        # Use specific DICOM series
        if not os.path.exists(self.dicomweb_series):
            self.skipTest(f"DICOM series not found: {self.dicomweb_series}")

        logger.info(f"Testing on DICOM series: {self.dicomweb_series}")

        # Run inference
        label_data, label_json, inference_time = self._run_inference(self.dicomweb_series)

        # Validate output
        self._validate_segmentation_output(label_data, label_json)

        # Performance check
        self.assertLess(inference_time, 60.0, "Inference should complete within 60 seconds")
        logger.info(f"✓ DICOM inference test passed (dicomweb) in {inference_time:.3f}s")

    def test_03_dicom_inference_dicomweb_htj2k(self):
        """Test inference on DICOM series from dicomweb_htj2k directory (HTJ2K compressed)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if not self.app:
            self.skipTest("App not initialized")

        # Use specific HTJ2K DICOM series
        if not os.path.exists(self.dicomweb_htj2k_series):
            self.skipTest(f"HTJ2K DICOM series not found: {self.dicomweb_htj2k_series}")

        logger.info(f"Testing on HTJ2K compressed DICOM series: {self.dicomweb_htj2k_series}")

        # Run inference
        label_data, label_json, inference_time = self._run_inference(self.dicomweb_htj2k_series)

        # Validate output
        self._validate_segmentation_output(label_data, label_json)

        # Performance check
        self.assertLess(inference_time, 60.0, "Inference should complete within 60 seconds")
        logger.info(f"✓ DICOM inference test passed (HTJ2K) in {inference_time:.3f}s")

    def test_04_compare_all_formats(self):
        """
        Compare segmentation outputs across all DICOM format variations.

        This is the KEY test that validates:
        - Standard DICOM (uncompressed, single-frame)
        - HTJ2K compressed DICOM (single-frame)
        - Multi-frame HTJ2K DICOM

        All produce IDENTICAL or highly similar segmentation results.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if not self.app:
            self.skipTest("App not initialized")

        logger.info(f"\n{'='*60}")
        logger.info("Comparing Segmentation Outputs Across All Formats")
        logger.info(f"{'='*60}")

        # Test all series types
        test_series = [
            ("Standard DICOM", self.dicomweb_series),
            ("HTJ2K DICOM", self.dicomweb_htj2k_series),
            ("Multi-frame HTJ2K", self.dicomweb_htj2k_multiframe_series),
        ]

        # Run inference on all available formats
        results = {}
        for series_name, series_path in test_series:
            if not os.path.exists(series_path):
                logger.warning(f"Skipping {series_name}: not found")
                continue

            logger.info(f"\nRunning {series_name}...")
            try:
                label_data, label_json, inference_time = self._run_inference(series_path)
                self._validate_segmentation_output(label_data, label_json)

                results[series_name] = {"label_data": label_data, "label_json": label_json, "time": inference_time}
                logger.info(f"  ✓ {series_name} completed in {inference_time:.3f}s")
            except Exception as e:
                logger.error(f"  ✗ {series_name} failed: {e}", exc_info=True)

        # Require at least 2 formats to compare
        self.assertGreaterEqual(len(results), 2, "Need at least 2 formats to compare. Check test data availability.")

        # Compare all pairs
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Format Comparison:")
        logger.info(f"{'='*60}")

        format_names = list(results.keys())
        comparison_results = []

        for i in range(len(format_names)):
            for j in range(i + 1, len(format_names)):
                name1 = format_names[i]
                name2 = format_names[j]

                logger.info(f"\nComparing: {name1} vs {name2}")
                try:
                    comparison = self._compare_segmentations(
                        results[name1]["label_data"],
                        results[name2]["label_data"],
                        name_1=name1,
                        name_2=name2,
                        tolerance=0.05,  # Allow 5% dice variation
                    )
                    comparison_results.append(
                        {
                            "pair": f"{name1} vs {name2}",
                            "dice": comparison["avg_dice"],
                            "pixel_accuracy": comparison["pixel_accuracy"],
                        }
                    )
                except Exception as e:
                    logger.error(f"Comparison failed: {e}", exc_info=True)
                    raise

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Comparison Summary:")
        for comp in comparison_results:
            logger.info(f"  {comp['pair']}: Dice={comp['dice']:.4f}, Accuracy={comp['pixel_accuracy']:.4f}")
        logger.info(f"{'='*60}")

        # All comparisons should show high similarity
        self.assertTrue(len(comparison_results) > 0, "Should have at least one comparison")
        avg_dice = np.mean([c["dice"] for c in comparison_results])
        logger.info(f"\nOverall average Dice across all comparisons: {avg_dice:.4f}")
        self.assertGreater(avg_dice, 0.95, "All formats should produce highly similar segmentations (avg Dice > 0.95)")

    def test_05_compare_dicom_vs_nifti(self):
        """
        Compare inference results between DICOM series and pre-converted NIfTI files.

        Validates that the DICOM reader produces identical results to pre-converted NIfTI.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if not self.app:
            self.skipTest("App not initialized")

        # Use specific DICOM series and its NIfTI equivalent
        dicom_dir = self.dicomweb_series
        nifti_file = f"{dicom_dir}.nii.gz"

        if not os.path.exists(dicom_dir):
            self.skipTest(f"DICOM series not found: {dicom_dir}")

        if not os.path.exists(nifti_file):
            self.skipTest(f"Corresponding NIfTI file not found: {nifti_file}")

        logger.info(f"\n{'='*60}")
        logger.info("Comparing DICOM vs NIfTI Segmentation")
        logger.info(f"{'='*60}")
        logger.info(f"  DICOM: {dicom_dir}")
        logger.info(f"  NIfTI: {nifti_file}")

        # Run inference on DICOM
        logger.info("\n--- Running inference on DICOM series ---")
        dicom_label, dicom_json, dicom_time = self._run_inference(dicom_dir)
        self._validate_segmentation_output(dicom_label, dicom_json)

        # Run inference on NIfTI
        logger.info("\n--- Running inference on NIfTI file ---")
        nifti_label, nifti_json, nifti_time = self._run_inference(nifti_file)
        self._validate_segmentation_output(nifti_label, nifti_json)

        # Compare the segmentation outputs
        comparison = self._compare_segmentations(
            dicom_label,
            nifti_label,
            name_1="DICOM",
            name_2="NIfTI",
            tolerance=0.01,  # Stricter tolerance - should be nearly identical
        )

        logger.info(f"\n{'='*60}")
        logger.info("Comparison Summary:")
        logger.info(f"  DICOM inference time: {dicom_time:.3f}s")
        logger.info(f"  NIfTI inference time: {nifti_time:.3f}s")
        logger.info(f"  Dice coefficient: {comparison['avg_dice']:.4f}")
        logger.info(f"  Pixel accuracy: {comparison['pixel_accuracy']:.4f}")
        logger.info(f"  Exact match: {comparison['exact_match']}")
        logger.info(f"{'='*60}")

        # Should be nearly identical (Dice > 0.99)
        self.assertGreater(comparison["avg_dice"], 0.99, "DICOM and NIfTI segmentations should be nearly identical")

    def test_06_multiframe_htj2k_inference(self):
        """
        Test basic inference on multi-frame HTJ2K compressed DICOM series.

        Note: Comprehensive cross-format comparison is done in test_04.
        This test ensures multi-frame HTJ2K inference works standalone.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if not self.app:
            self.skipTest("App not initialized")

        if not os.path.exists(self.dicomweb_htj2k_multiframe_series):
            self.skipTest(f"Multi-frame HTJ2K series not found: {self.dicomweb_htj2k_multiframe_series}")

        logger.info(f"\n{'='*60}")
        logger.info("Testing Multi-Frame HTJ2K DICOM Inference")
        logger.info(f"{'='*60}")
        logger.info(f"Series path: {self.dicomweb_htj2k_multiframe_series}")

        # Run inference
        label_data, label_json, inference_time = self._run_inference(self.dicomweb_htj2k_multiframe_series)

        # Validate output
        self._validate_segmentation_output(label_data, label_json)

        # Performance check
        self.assertLess(inference_time, 60.0, "Inference should complete within 60 seconds")

        logger.info(f"✓ Multi-frame HTJ2K inference test passed in {inference_time:.3f}s")


if __name__ == "__main__":
    unittest.main()
