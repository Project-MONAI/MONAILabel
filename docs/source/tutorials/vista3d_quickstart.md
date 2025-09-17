# VISTA3D Quick Start Guide

This quick start guide helps you get VISTA3D running with MONAI Label in minutes.

## What is VISTA3D?

VISTA3D is a universal foundation model for 3D CT segmentation that can segment 117+ anatomical structures including organs, bones, vessels, and soft tissues. It supports both automatic and interactive segmentation workflows.

## Quick Setup

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM
- MONAI Label installed

### 2. Download and Start
```bash
# Download the bundle app
monailabel apps --download --name monaibundle --output apps

# Start VISTA3D server
monailabel start_server \
  --app apps/monaibundle \
  --studies /path/to/ct/volumes \
  --conf models vista3d \
  --conf preload true
```

### 3. Connect Viewer

**For OHIF Viewer:**
```bash
cd apps/monaibundle
# Follow OHIF setup instructions in main tutorial
```

**For 3D Slicer:**
1. Install MONAI Label extension
2. Connect to server at `http://localhost:8000`
3. VISTA3D models will appear automatically

## Basic Usage

### Automatic Segmentation
1. Load CT volume in your viewer
2. Select VISTA3D model
3. Run auto-segmentation for 117+ structures

### Interactive Prompting
1. Choose target anatomy (e.g., liver, kidney)
2. Click positive points on the structure
3. Right-click negative points (optional)
4. Generate refined segmentation

### Class Selection
1. Choose anatomical categories:
   - Organs (liver, spleen, kidneys, etc.)
   - Bones (vertebrae, ribs, pelvis, etc.)
   - Vessels (aorta, vena cava, etc.)
2. Run targeted segmentation

## Common Use Cases

- **Clinical**: Multi-organ analysis, surgical planning
- **Research**: Anatomical quantification, population studies
- **AI Development**: Training data generation, model evaluation

## Need Help?

- 📖 [Complete VISTA3D Tutorial](vista3d_tutorial.md)
- 🐛 [Report Issues](https://github.com/Project-MONAI/MONAILabel/issues)
- 💬 [Community Support](https://projectmonai.slack.com/)

## Next Steps

- Explore advanced configuration options
- Set up automated workflows
- Integrate with clinical systems
- Customize for specific use cases