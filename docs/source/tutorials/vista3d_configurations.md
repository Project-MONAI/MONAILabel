# VISTA3D Configuration Examples

This document provides sample configurations for integrating VISTA3D with MONAI Label.

## Basic VISTA3D Server Configuration

### 1. Standard Setup
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf preload true \
  --conf skip_trainers true
```

### 2. Memory-Optimized Setup
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf vista3d_patch_size "[64,64,64]" \
  --conf vista3d_overlap 0.125 \
  --conf vista3d_cpu_fallback true
```

### 3. High-Performance Setup
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf vista3d_patch_size "[128,128,128]" \
  --conf vista3d_overlap 0.5 \
  --conf preload true \
  --conf workers 4
```

## VISTA3D with DICOM Web

### 4. PACS Integration
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies "http://pacs.hospital.com/dicom-web" \
  --studies_user "username" \
  --studies_password "password" \
  --conf models vista3d \
  --conf preload true
```

## Custom VISTA3D Bundle Configuration

### 5. Custom Bundle Path
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models "vista3d:/path/to/custom/vista3d/bundle" \
  --conf vista3d_prompting true
```

## Multi-Model Setup

### 6. VISTA3D with Other Models
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models "vista3d,spleen_ct_segmentation,swin_unetr_btcv_segmentation" \
  --conf preload true
```

## VISTA3D with Interactive Features

### 7. Full Interactive Setup
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf vista3d_prompting true \
  --conf vista3d_point_prompts true \
  --conf vista3d_class_prompts true \
  --conf preload true
```

## Development and Testing

### 8. Debug Configuration
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf debug true \
  --conf vista3d_cache_transforms false \
  --log_level DEBUG
```

### 9. Performance Profiling
```bash
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d \
  --conf profile true \
  --conf vista3d_profile_inference true
```

## Docker Deployment

### 10. Docker with VISTA3D
```bash
docker run --gpus all --rm -ti \
  -p 8000:8000 \
  -v /data/ct_volumes:/data/studies \
  -v /data/models:/data/models \
  projectmonai/monailabel:latest \
  monailabel start_server \
    --app /opt/monailabel/sample-apps/monaibundle \
    --studies /data/studies \
    --conf models vista3d \
    --conf preload true \
    --host 0.0.0.0
```

## Configuration Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `vista3d_patch_size` | Input patch size for inference | `[96,96,96]` | `"[64,64,64]"` |
| `vista3d_overlap` | Sliding window overlap ratio | `0.25` | `0.5` |
| `vista3d_prompting` | Enable interactive prompting | `false` | `true` |
| `vista3d_point_prompts` | Enable point-based prompting | `true` | `false` |
| `vista3d_class_prompts` | Enable class-based prompting | `true` | `false` |
| `vista3d_cpu_fallback` | Use CPU when GPU memory insufficient | `false` | `true` |
| `vista3d_cache_transforms` | Cache preprocessing transforms | `true` | `false` |
| `vista3d_profile_inference` | Profile inference performance | `false` | `true` |

## Environment Variables

```bash
# Set GPU memory fraction
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set cache directories
export MONAI_CACHE_DIR=/data/cache
export VISTA3D_CACHE_DIR=/data/vista3d_cache

# Start server with environment
monailabel start_server \
  --app apps/monaibundle \
  --studies datasets/ct_volumes \
  --conf models vista3d
```

## Troubleshooting Common Configurations

### GPU Memory Issues
```bash
# Reduce memory usage
--conf vista3d_patch_size "[48,48,48]"
--conf vista3d_overlap 0.125
--conf vista3d_cpu_fallback true
```

### Network/Download Issues
```bash
# Use local model cache
--conf zoo_source local
--conf model_path /local/path/to/vista3d

# Use alternative zoo repo
--conf zoo_repo "custom-repo/model-zoo"
```

### Performance Optimization
```bash
# Optimize for speed
--conf vista3d_patch_size "[128,128,128]"
--conf vista3d_overlap 0.5
--conf workers 8
--conf preload true
```