---
name: monailabel-radiology
description: Annotate CT/MRI in Slicer/OHIF; start/stop interactive models, switch point/box modes, edit spatial hints, localize anatomy, clear masks, change colors, undo/redo or submit.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: set_interaction_mode annotate locate_region remove_regions clear_segments set_label_color edit_spatial_prompts viewer_edit open_viewer
metadata:
  monailabel-context: viewer
  monailabel-sources: volume3d
---

# Radiology

“Clear the current slice for spleen” clears the segmentation through clear_segments. Only requests naming hints, points or boxes use edit_spatial_prompts. When clearing plural hints without a named label, set all_targets=true; the active interaction target does not restrict that request. “Whole volume” changes scope only: preserve the requested target, input type and polarity. A named organ requires target; omit all_targets or set it false. Call edit_spatial_prompts for these examples:

- “Clear all spatial hints”: operation=clear, kind=all, polarity=all, scope=full, all_targets=true.
- “Clear negative points on this slice”: operation=clear, kind=point, polarity=negative, scope=current_slice, all_targets=true.
- “Clear spleen boxes in the whole volume”: operation=clear, kind=box, polarity=all, scope=full, target=Spleen.


If context.video is present, load monailabel-video and use its tools instead; this also applies to single-frame segmentation with trained U-Nets. Radiology tools require medical image/volume context.

Use current_slice for “this slice”, with current native geometry. Explicit full volume/all slices means scope=full even with an active selection. A selected_region is exactly one crop, never tiled. Never infer slice geometry from history. VISTA3D supports CT anatomy, not MRI/pathology. GPT-6 Astra is the project default when available and uses windowed slices. Honor an explicit VISTA3D request for CT annotation; named models override the default.

“Annotate the spleen on the current slice using GPT Astra” means annotate(targets=["Spleen"], scope="current_slice", model_name="GPT-6 Astra"). Pass the available Astra name even if context.model_id currently selects VISTA3D. Never copy the selected ID over an explicitly requested model.

locate_region(kind=box) creates an editable outline, not a mask. kind=roi uses 1-based inclusive first/last slices. remove_regions removes legacy localization ROIs; interactive input boxes use edit_spatial_prompts. clear_segments changes only named mask labels; use all_targets=true only for explicit all-segment requests. Default clearing scope is full; “on this slice” remains current_slice even when “all” refers to labels. Never expand an unsupported/missing slice or region to full volume. Saved revisions and label definitions remain. Undo requires advertised viewer support.

set_label_color updates shared display colors, not voxels. Convert explicit named colors to RGB hex; color=anatomical restores Slicer Generic Anatomy defaults (not a universal DICOM organ palette). Omit project-creation colors unless requested.

SAM 2.1 annotates one object in a 2D image/selected slice; MedSAM2 can propagate a seed through a full medical volume. They need a current box/positive points; negatives refine the object. A QuPath selected image_region can provide the box. Name one target. These models do not localize organs by name or separate every nucleus in a broad region. Never tile an object prompt. SAM training and unprompted evaluation are unavailable.

nnInteractive annotates one prompted object in CT/MRI with native 3D inference. It uses positive/negative points across slices and a box drawn on one slice. It needs viewer hints, never an organ name alone or invented coordinates. Use scope=full for the whole volume; current_slice changes only that slice. It cannot annotate an unprompted batch. Fine-tuning and unprompted evaluation are unavailable.

For viewers advertising set_interaction_mode, “Start nnInteractive for spleen” activates mode=positive, target=Spleen, model_name=nnInteractive. This selects the model and waits for native user clicks; do not run inference yet. “Switch to negative points” retains the active target/model. In interaction mode, “Switch to liver” activates positive mode with target=Liver, retaining the model and each label’s existing hints. “Draw a spleen box” activates mode=box, target=Spleen without coordinates. “Stop interaction mode” uses mode=navigate and retains hints/masks. Distinguish entering drawing mode from creating a box at explicit coordinates, which uses edit_spatial_prompts.

Use edit_spatial_prompts to add a box or point at explicit coordinates, move/resize an existing hint, or clear hints. It does not activate drawing: “Draw a spleen box” without coordinates uses set_interaction_mode(mode=box, target=Spleen), then the user draws it. Current spatial_objects are authoritative, including after a possibly rejected edit. kind=box or point; operation=move for moving/resizing an existing object, never add another. Explicit user coordinates are zero-based source voxels; two values use axes other than slice.axis. Ask for zero-based input if explicitly one-based. Never invent coordinates. box_center=true is allowed only for an explicitly requested editable starting point inside a box, not anatomical localization. Moving an ambiguous object requires clarification. Anatomical localization without coordinates uses locate_region with an explicitly chosen capable model.


After an edit, wait for fresh native geometry before a subsequent annotation request uses the hints. Only advertise supported viewer actions.

To open or view the sample, use open_viewer with the requested viewer (slicer or ohif). Opening a viewer does not request annotation; do not call annotate for a viewing request.
