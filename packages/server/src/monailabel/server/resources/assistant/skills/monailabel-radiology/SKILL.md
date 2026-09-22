---
name: monailabel-radiology
description: Segment structures in one medical image, full volume or selected slice in Slicer or OHIF. Also edit SAM boxes and points, localize structures, clear masks or outlines, and adjust anatomy colors.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: annotate locate_region remove_regions clear_segments set_label_color edit_spatial_prompts viewer_edit open_viewer
metadata:
  monailabel-context: viewer
---

# Radiology

Use current_slice for “this slice”, with current native geometry. Explicit full volume/all slices means scope=full even with an active selection. A selected_region is exactly one crop, never tiled. Never infer slice geometry from history. VISTA3D supports CT anatomy, not MRI/pathology. VISTA3D remains the default for supported CT targets. Hosted alternatives such as GPT-6 Astra use windowed slices; explicit model choices take precedence.

“Annotate the spleen on the current slice using GPT Astra” means annotate(targets=["Spleen"], scope="current_slice", model_name="GPT-6 Astra"). Pass the available Astra name even if context.model_id currently selects VISTA3D. Never copy the selected ID over an explicitly requested model.

locate_region(kind=box) creates an editable outline, not a mask. kind=roi uses 1-based inclusive first/last slices. remove_regions removes outlines. clear_segments changes only named mask labels; use all_targets=true only for explicit all-segment requests. Default clearing scope is full; “on this slice” remains current_slice even when “all” refers to labels. Never expand an unsupported/missing slice or region to full volume. Saved revisions and label definitions remain. Undo requires advertised viewer support.

set_label_color updates shared display colors, not voxels. Convert explicit named colors to RGB hex; color=anatomical restores Slicer Generic Anatomy defaults (not a universal DICOM organ palette). Omit project-creation colors unless requested.

SAM 2.1 annotates one object in a 2D image/selected slice; MedSAM2 can propagate a seed through a full medical volume. They need a current box/positive points; negatives refine the object. A QuPath selected image_region can provide the box. Name one target. These models do not localize organs by name or separate every nucleus in a broad region. Never tile an object prompt. SAM training and unprompted evaluation are unavailable.

For viewers advertising edit_spatial_prompts, use that tool for native boxes and points. Current spatial_objects are authoritative, including after a possibly rejected edit. kind=box or point; operation=move for moving/resizing an existing object, never add another. Explicit user coordinates are zero-based source voxels; two values use axes other than slice.axis. Ask for zero-based input if explicitly one-based. Never invent coordinates. box_center=true is allowed only for an explicitly requested editable starting point inside a box, not anatomical localization. Missing coordinates or ambiguous/multiple objects require clarification. Anatomical localization without coordinates uses locate_region with an explicitly chosen capable model.

Clear hints through edit_spatial_prompts, never clear_segments: negative points on this slice means kind=point, polarity=negative, scope=current_slice, all_targets=true unless named. All SAM hints on this slice means kind=all, polarity=all, all_targets=true. A spleen box means kind=box, target=Spleen, polarity=all. After an edit, wait for fresh native geometry before a subsequent annotate request runs SAM. Only advertise supported viewer actions.

To open or view the sample, use open_viewer with the requested viewer (slicer or ohif). Opening a viewer does not request annotation; do not call annotate for a viewing request.
