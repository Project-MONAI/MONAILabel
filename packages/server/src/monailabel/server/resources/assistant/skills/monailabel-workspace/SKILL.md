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
