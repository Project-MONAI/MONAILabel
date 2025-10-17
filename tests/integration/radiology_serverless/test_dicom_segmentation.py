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
        "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266"
    )
    dicomweb_htj2k_series = os.path.join(
        data_dir,
        "dataset",
        "dicomweb_htj2k",
        "e7567e0a064f0c334226a0658de23afd",
        "1.2.826.0.1.3680043.8.274.1.1.8323329.686521.1629744176.620266"
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
                import nibabel as nib
                nii = nib.load(label_data)
                array = nii.get_fdata()
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
    
    def test_04_dicom_inference_both_formats(self):
        """Test inference on both standard and HTJ2K compressed DICOM series."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        if not self.app:
            self.skipTest("App not initialized")
        
        # Test both series types
        test_series = [
            ("Standard DICOM", self.dicomweb_series),
            ("HTJ2K DICOM", self.dicomweb_htj2k_series),
        ]
        
        total_time = 0
        successful = 0
        
        for series_type, dicom_dir in test_series:
            if not os.path.exists(dicom_dir):
                logger.warning(f"Skipping {series_type}: {dicom_dir} not found")
                continue
            
            logger.info(f"\nProcessing {series_type}: {dicom_dir}")
            
            try:
                label_data, label_json, inference_time = self._run_inference(dicom_dir)
                self._validate_segmentation_output(label_data, label_json)
                
                total_time += inference_time
                successful += 1
                logger.info(f"✓ {series_type} success in {inference_time:.3f}s")
                
            except Exception as e:
                logger.error(f"✗ {series_type} failed: {e}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary: {successful}/{len(test_series)} series processed successfully")
        if successful > 0:
            logger.info(f"Total inference time: {total_time:.3f}s")
            logger.info(f"Average time per series: {total_time/successful:.3f}s")
        logger.info(f"{'='*60}")
        
        # At least one should succeed
        self.assertGreater(successful, 0, "At least one DICOM series should be processed successfully")
    
    def test_05_compare_dicom_vs_nifti(self):
        """Compare inference results between DICOM series and pre-converted NIfTI files."""
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
        
        logger.info(f"Comparing DICOM vs NIfTI inference:")
        logger.info(f"  DICOM: {dicom_dir}")
        logger.info(f"  NIfTI: {nifti_file}")
        
        # Run inference on DICOM
        logger.info("\n--- Running inference on DICOM series ---")
        dicom_label, dicom_json, dicom_time = self._run_inference(dicom_dir)
        
        # Run inference on NIfTI
        logger.info("\n--- Running inference on NIfTI file ---")
        nifti_label, nifti_json, nifti_time = self._run_inference(nifti_file)
        
        # Validate both
        self._validate_segmentation_output(dicom_label, dicom_json)
        self._validate_segmentation_output(nifti_label, nifti_json)
        
        logger.info(f"\nPerformance comparison:")
        logger.info(f"  DICOM inference time: {dicom_time:.3f}s")
        logger.info(f"  NIfTI inference time: {nifti_time:.3f}s")
        
        # Both should complete successfully
        self.assertIsNotNone(dicom_label, "DICOM inference should succeed")
        self.assertIsNotNone(nifti_label, "NIfTI inference should succeed")


if __name__ == "__main__":
    unittest.main()

