---
name: monailabel-workspace
description: Create projects, list models or activity, open credential, model or other setup forms, launch viewers, choose the default annotation model, select cases, or cancel jobs.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: open_form create_project select_annotation_model open_viewer cancel_job next_cases
metadata:
  monailabel-context: workspace
---

# Workspace

Create projects with the user's exact name; labels are optional. Without a name, open_form(project); never invent one. Missing setup details use forms. Local files use open_form(dataset); DICOM uses open_form(dicom); secrets use open_form(credential). Opening a form or viewer does not prove completion.

Inspect workspace to resolve names/IDs. “Use Astra from now on” changes selection; “annotate using Astra” belongs to an annotation skill. next_cases only ranks/selects images; it does not segment or submit them. “Next case” means limit=1.

For undo, redo or other edits in an open viewer, load its annotation skill: monailabel-video when context.video is present, monailabel-pathology for image_region, or monailabel-radiology for a volume/slice. Then use the advertised viewer action; no additional viewer selection or edit description is needed for undo/redo.

For training defaults, inspect_workspace(collection=training_recipes, recipe_id=the requested recipe). These are actual hyperparameters; action arguments such as start_now are not training settings. Inspect learners for a named setup's saved config. Do not guess defaults from skill prose.
