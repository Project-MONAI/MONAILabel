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

Only import when the user requests data. Creating a model for already imported labels uses monailabel-model-training; it does not request another import.

The attached workspace data lists available dataset templates. Resolve the user's dataset name to its template_id there; inspect_workspace(dataset_templates) can provide more details. Never ask the user for a catalog ID or local path. Use import_dataset_template for one portion. Default five images for annotation (split=pool), labels only if requested. Evaluation (split=validation) always includes supplied labels. For this single-portion tool, “all” sets all_samples=true. Source test sections have no labels. Honor counts, modality and structures; TotalSegmentator needs explicit structures. Browsing does not import.

Combined percentage requests use import_dataset_split. This tool imports ALL labeled source cases by default: omit limit or set it to null. Both import tools also accept all_samples=true for an explicit request for all cases. Set a numeric limit only for a user-specified count. One job imports BOTH portions with split=pool; evaluation_percentage reserves independent evaluation cases, separate from model validation percentages.

Set include_masks from the requested annotation/training content:

- “Import with labels” or “import labeled cases; reserve 20% for evaluation” → true. Keep the supplied training references as well as the evaluation references.
- “80% images only, 20% with evaluation labels” → false. Only evaluation receives supplied references.

Evaluation cases always receive their supplied labels and are accepted on import. Annotation/training labels remain pending review. Imports reuse cached archives and protect existing annotations.

Video templates import one original clip for tool tracking in CVAT, with instrument classes and no reference tracks. Use split=pool, include_masks=false and offset=0; image training and evaluation options do not apply. The HyperKvasir tool-tracking sample contains a moving snare. The imported clip appears alongside images in the shared Datasets table, with a CVAT action. Repeating an import preserves submitted tracks and external CVAT drafts.

Examples:

- “Import 10 spleen images” → inspect the catalog, then import_dataset_template with limit=10 and include_masks=false.
- “Use 75% for annotation and 25% with labels for evaluation” → import_dataset_split with evaluation_percentage=25 and include_masks=false; omit limit unless the user specifies a total.
- “Import 5 evaluation samples” → import_dataset_template with limit=5, split=validation and include_masks=true.

Local files use open_form(dataset); DICOM uses open_form(dicom). Do not ask for a path when the user names a catalog dataset.

For nnU-Net prostate-zone training, use Task05_Prostate with channel=0 for T2. Keep its original PZ/TZ labels. The prostate-mri template merges both zones into a whole-gland mask for TotalSegmentator and is a different target. The CT lung-tumor example uses Task06_Lung with its original labels; liver/tumor training uses Task03_Liver. Never substitute whole-organ masks for tumor targets.
