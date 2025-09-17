# VISTA3D Tutorial for MONAI Label

VISTA3D (Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography) is an advanced foundation model for 3D medical image segmentation that can segment a wide variety of anatomical structures in CT volumes.

## Overview

VISTA3D is designed to work as a universal segmentation model that can handle multiple anatomical structures simultaneously. It supports both automatic segmentation and interactive segmentation with point prompts, making it highly versatile for clinical workflows.

### Key Features

- **Universal Segmentation**: Supports 117+ anatomical classes including organs, bones, vessels, and tumors
- **Interactive Prompting**: Point-based interactive segmentation for precise control
- **Class-based Prompting**: Segment specific anatomical structures by class selection
- **3D Native**: Designed specifically for 3D CT volumes
- **High Performance**: State-of-the-art segmentation accuracy across diverse anatomy

## Supported Viewers

VISTA3D integration is currently available through:

- **OHIF Viewer**: Web-based DICOM viewer with full VISTA3D support
- **3D Slicer**: Desktop application with VISTA3D capabilities (requires bundle integration)

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Python**: Python 3.8+ with MONAI Label installed
- **Storage**: Several GB for model weights

### Software Installation

1. Install MONAI Label with VISTA3D dependencies:
```bash
pip install monailabel[vista3d]
```

2. For development/latest features:
```bash
git clone https://github.com/Project-MONAI/MONAILabel
cd MONAILabel
pip install -e .
```

## Data Preparation

VISTA3D works with standard CT DICOM series or NIfTI volumes. Prepare your data as follows:

### For Local Datastore
```
datasets/
├── ct_volume_001.nii.gz
├── ct_volume_002.nii.gz
└── ...
```

### For DICOM Web
Ensure your PACS/DICOM server supports DICOMWeb standard for direct integration.

## Setting Up VISTA3D

### Method 1: Using MONAIBundle App (Recommended)

The VISTA3D model can be integrated through the MONAI Bundle application:

```bash
# Download the monaibundle app
monailabel apps --download --name monaibundle --output apps

# Start server with VISTA3D model (when available in model zoo)
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d_v1.0.0 \
  --conf preload true
```

### Method 2: Custom VISTA3D Configuration

For advanced users wanting to customize VISTA3D behavior:

1. Create a custom configuration in your bundle app:
```python
# lib/configs/vista3d_config.py
from monailabel.tasks.infer.bundle import BundleInferTask

class VISTA3DConfig:
    def __init__(self):
        self.model_name = "vista3d"
        self.bundle_path = "path/to/vista3d/bundle"
        self.labels = {
            "liver": 1,
            "spleen": 2,
            "kidney_left": 3,
            "kidney_right": 4,
            # ... additional anatomical labels
        }
```

## Usage Examples

### 1. Automatic Segmentation

Start the server and use the Auto Segmentation feature:

```bash
# Start MONAI Label server
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d_v1.0.0

# Open OHIF viewer and navigate to:
# http://localhost:3000/viewer?StudyInstanceUIDs=<study_id>
```

In the OHIF viewer:
1. Load a CT volume
2. Open MONAI Label plugin panel
3. Select "Auto Segmentation" tab
4. Choose VISTA3D model
5. Click "Run Segmentation" for automatic multi-organ segmentation

### 2. Interactive Point Prompting

For targeted segmentation with point prompts:

1. Load CT volume in OHIF viewer
2. Select "Point Prompts" tab in MONAI Label panel
3. Choose VISTA3D model
4. Select target anatomical structure from dropdown
5. Click on the structure in the image to add positive points
6. Right-click to add negative points (background)
7. Click "Run Inference" to generate segmentation

### 3. Class-based Prompting

For segmenting specific anatomical categories:

1. Select "Class Prompts" tab
2. Choose VISTA3D model
3. Select anatomical categories:
   - **Organs**: Liver, spleen, kidneys, etc.
   - **Bones**: Vertebrae, ribs, pelvis, etc.
   - **Vessels**: Aorta, vena cava, pulmonary vessels, etc.
4. Click "Run Inference" to segment selected classes

## Advanced Configuration

### Model Parameters

VISTA3D supports various inference parameters that can be configured:

```bash
# Start with custom parameters
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets \
  --conf models vista3d_v1.0.0 \
  --conf vista3d_patch_size "[96,96,96]" \
  --conf vista3d_overlap 0.25 \
  --conf vista3d_batch_size 1
```

### Memory Optimization

For systems with limited GPU memory:

```bash
# Use smaller patch sizes and enable CPU fallback
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets \
  --conf models vista3d_v1.0.0 \
  --conf vista3d_patch_size "[64,64,64]" \
  --conf vista3d_cpu_fallback true
```

## Viewer-Specific Instructions

### OHIF Viewer Setup

1. **Install OHIF with MONAI Label Extension**:
```bash
# Clone OHIF with MONAI Label extension
git clone https://github.com/Project-MONAI/MONAILabel
cd MONAILabel/plugins/ohifv3

# Install dependencies and start
npm install
npm run dev
```

2. **Configure OHIF for VISTA3D**:
   - The OHIF extension automatically detects VISTA3D models
   - Point prompts and class prompts are available in the MONAI Label panel
   - Auto segmentation supports multi-organ prediction

### 3D Slicer Integration

For 3D Slicer users:

1. **Install MONAI Label Extension**:
   - Open 3D Slicer
   - Go to Extension Manager
   - Search and install "MONAI Label"

2. **Connect to VISTA3D Server**:
   - Open MONAI Label module
   - Set server URL: `http://localhost:8000`
   - VISTA3D models will appear in the model selection

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   - Reduce patch size: `--conf vista3d_patch_size "[48,48,48]"`
   - Enable CPU inference: `--conf vista3d_device cpu`

2. **Model Loading Failures**:
   - Verify VISTA3D bundle is correctly downloaded
   - Check network connectivity for model zoo access
   - Ensure sufficient disk space for model weights

3. **Slow Inference**:
   - Verify GPU is being used: check `nvidia-smi`
   - Optimize batch size based on available GPU memory
   - Consider model quantization for faster inference

### Performance Tips

- **Preload Models**: Use `--conf preload true` for faster first inference
- **Batch Processing**: Process multiple volumes in sequence for efficiency
- **ROI Selection**: Use viewer tools to define regions of interest for faster processing

## Example Workflows

### Clinical Workflow: Abdominal Organ Segmentation

1. Load abdominal CT scan in OHIF
2. Select VISTA3D model
3. Use "Class Prompts" to select:
   - Liver
   - Spleen  
   - Left/Right Kidneys
   - Pancreas
4. Run segmentation and review results
5. Use point prompts to refine any inaccurate regions
6. Export segmentations for clinical use

### Research Workflow: Comprehensive Anatomy Analysis

1. Load full-body CT scan
2. Run automatic segmentation with VISTA3D
3. Review 117+ automatically segmented structures
4. Use interactive prompts to add/remove structures as needed
5. Export detailed anatomical segmentations for analysis

## References

- [VISTA3D Paper](https://arxiv.org/abs/2406.05285): Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography
- [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html)
- [MONAI Label Documentation](https://docs.monai.io/projects/label/en/latest/)

## Support

For VISTA3D-specific issues:
- Check [MONAI Label GitHub Issues](https://github.com/Project-MONAI/MONAILabel/issues)
- Join [MONAI Slack Community](https://projectmonai.slack.com/)
- Review [MONAI Label Discussions](https://github.com/Project-MONAI/MONAILabel/discussions)