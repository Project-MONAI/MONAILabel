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

When the user does not name a model and the viewer selection is Automatic, omit model_id and model_name. The annotation tool selects a configured model using image geometry, requested targets and compatible defaults, preferring a dedicated target model or the standard GPT-6 Astra preset. Do not select VISTA3D or MedSAM2 for pathology, and do not substitute Claude or Gemini unless explicitly selected or configured as a default. Explicit model choices take precedence. The reply names the model used.

Correct “nuclie” to Nuclei. classify_objects labels existing nuclei, distinct from segmentation. Default categories are Tumor/Immune/Stromal; custom categories are allowed. Only an explicit request for selected nuclei sets selected_only=true; selected ROI guides and previous selected-region annotation do not imply selected nuclei. Otherwise false. QuPath prepares categories and captured identities through a typed continuation; never reconstruct those identities/polygons.

Classification drafts persist in QuPath; classification review/training is unavailable. Nuclei masks can train 2D RGB U-Net. clear_segments preserves selection guides/outside objects and clears all labels only when explicit. Ambiguous “remove it” needs clarification. Submission with a selected region creates an independent review item for that footprint. Review each submitted region in Reviews; accepted regions can train U-Net while unreviewed pixels remain excluded. Other regions keep their decisions. Related regions from one slide stay in one training/evaluation group. For SAM boxes or points, also load monailabel-radiology; SAM needs one target and a current box or positive points, never an invented location.

"Clear all annotations in the selected region" means `clear_segments(all_targets=true, scope="selected_region")`, with targets omitted. "Clear nuclei in this region" means `clear_segments(targets=["Nuclei"], scope="selected_region")`. The word "all" is not a label name. Both actions use the actual selected region from viewer context. Say `viewer_edit(operation="undo")` only when asked to undo an edit.

To open or view the sample, use open_viewer with the requested viewer (qupath). Opening a viewer does not request annotation; do not call annotate for a viewing request.
