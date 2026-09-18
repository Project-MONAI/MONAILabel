---
name: monailabel-dataset-import
description: Import sample datasets, local images and labels, video tool-tracking samples, or DICOM data. Use for counts, source splits, and combined annotation/evaluation imports.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: open_form import_dataset_template import_dataset_split
metadata:
  monailabel-context: project
  monailabel-collections: dataset_templates
---

# Dataset import

The attached workspace data lists available dataset templates. Resolve the user's dataset name to its template_id there; inspect_workspace(dataset_templates) can provide more details. Never ask the user for a catalog ID or local path. Use import_dataset_template for one portion. Default five images for annotation (split=pool), labels only if requested. Evaluation (split=validation) always includes supplied labels. “All” sets all_samples=true. Source test sections have no labels. Honor counts, modality and structures; TotalSegmentator needs explicit structures. Browsing does not import.

Combined percentage requests use import_dataset_split: 80% images only for annotation and 20% images+labels for evaluation means evaluation_percentage=20, include_masks=false. Omit limit unless a total count was requested: percentages alone use the entire labeled source section. One job imports BOTH portions; never substitute five evaluation cases. This is independent of model validation percentages. Imports reuse cached archives, protect existing annotations and leave imported labels pending review.

Video templates import one original clip for tool tracking in CVAT, with instrument classes and no reference tracks. Use split=pool, include_masks=false and offset=0; image training and evaluation options do not apply. The HyperKvasir tool-tracking sample contains a moving snare. The imported clip appears alongside images in the shared Datasets table, with a CVAT action. Repeating an import preserves submitted tracks and external CVAT drafts.

Examples:

- “Import 10 spleen images” → inspect the catalog, then import_dataset_template with limit=10 and include_masks=false.
- “Use 75% for annotation and 25% with labels for evaluation” → import_dataset_split with evaluation_percentage=25 and include_masks=false; omit limit unless the user specifies a total.
- “Import 5 evaluation samples” → import_dataset_template with limit=5, split=validation and include_masks=true.

Local files use open_form(dataset); DICOM uses open_form(dicom). Do not ask for a path when the user names a catalog dataset.
