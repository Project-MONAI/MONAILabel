---
name: monailabel-pathology
description: Segment nuclei or other structures in one pathology image, whole image or selected region in QuPath. Classify existing nuclei into cell categories or edit pathology drafts.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: annotate classify_objects remove_regions clear_segments set_label_color viewer_edit open_viewer
metadata:
  monailabel-context: viewer
---

# Pathology

Bounded RGB images use QuPath. Selected area means annotate(scope=selected_region), one crop without tiling, using viewer geometry. Explicit whole image/slide means scope=full regardless of selection; optional tile_size defaults to 256. Gigapixel streaming is unavailable.

Correct “nuclie” to Nuclei. classify_objects labels existing nuclei, distinct from segmentation. Default categories are Tumor/Immune/Stromal; custom categories are allowed. Only an explicit request for selected nuclei sets selected_only=true; selected ROI guides and previous selected-region annotation do not imply selected nuclei. Otherwise false. QuPath prepares categories and captured identities through a typed continuation; never reconstruct those identities/polygons.

Classification drafts persist in QuPath; classification review/training is unavailable. Nuclei masks can train 2D RGB U-Net. clear_segments preserves selection guides/outside objects and clears all labels only when explicit. Ambiguous “remove it” needs clarification. A partial-region proposal is not a complete annotation: inspect the entire imported field before submission. For SAM boxes or points, also load monailabel-radiology; SAM needs one target and a current box or positive points, never an invented location.

To open or view the sample, use open_viewer with the requested viewer (qupath). Opening a viewer does not request annotation; do not call annotate for a viewing request.
