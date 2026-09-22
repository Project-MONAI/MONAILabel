# Datasets, models and learning

## Import data

In **Datasets → Import files**, choose **Annotation & training** for images you want to annotate and use for learning. Import images alone or images with labels. Choose **Evaluation only** for independently labeled images that must stay out of every model's training, and give the evaluation set a name.

Image/label pairs match by filename; labels may have `_mask`, `_label`, `_labels`, `_seg` or `_segmentation` suffixes. Map foreground values under **Structures in these labels**; zero is background. Confirm **These labels have been reviewed** only when true. Otherwise they enter Reviews as pending.

Use **Advanced options** to group related images by patient or slide and record label provenance. Related groups and exact decoded-image duplicates stay together. Reimporting never overwrites different existing annotations. Partial failures keep successful imports and identify the cases to retry.

Supported inputs are scalar NIfTI and bounded PNG/JPEG/TIFF/SVS. Labels must be integer NIfTI, PNG or TIFF maps with matching geometry. Color masks, polygons and XML need conversion first. Limits are 128 MiB per source file, 256 MiB decompressed NIfTI, 67,108,864 voxels and 16,777,216 image pixels. Projects allow 31 foreground structures. See [viewers](viewers.md#dicom-import-and-viewing) for DICOM import.

## Sample datasets

**Datasets → Sample datasets** places the source beside **What to import**. Choose images only, images with labels, or unlabeled test images where available. The default imports five samples; **All samples** and **Advanced options** provide other counts. Evaluation imports always include labels. Public labels need review.

| Source | Available import |
| --- | --- |
| Medical Decathlon | All ten tasks; labeled training or unlabeled test images. Select one scalar modality for multichannel tasks. |
| TotalSegmentator CT | Pinned v2.0.1 sample or full archive; optional masks for selected structures. |
| OpenSlide | Small Aperio SVS region for trying QuPath. |
| MoNuSeg, CAMELYON, BACH | Source links and download guidance; automatic adapters are not available. |

The source's “training” section can supply your evaluation images and labels. Decathlon's test section has no supplied labels. A slide's crops are not independent patients or validation cases.

```text
Import 10 spleen images from Medical Decathlon to start annotation
Import 5 samples for evaluation from Decathlon spleen dataset
Import 80% of Decathlon spleen images for annotation and 20% with labels for evaluation
```

A combined percentage request imports both portions in one job. With no count, it uses the whole labeled section: 41 Spleen cases become 32 annotation images and 9 evaluation pairs. Evaluation rounds up to whole cases. This choice is separate from each model's training/validation split.

Archives are verified and reused from **workspace/.cache/datasets/**, even when sample counts or label choices change. A small sample still downloads the full archive. Interrupted partial downloads restart. `MONAILABEL_DATASETS_DIR` overrides the archive location; `MONAILABEL_CACHE_DIR` overrides the cache parent. Project deletion keeps downloads; clearing the whole workspace removes them.

## Review annotations

Inspect and correct annotations in [Slicer, QuPath or OHIF](viewers.md), then submit. Submission creates a pending review. In **Reviews**, select rows and use **Review decision** to mark them Good, Needs changes or Pending without reopening the viewer. The header checkbox selects the current page; selections persist across pages.

## Create and manage models

**Models → Add model** connects a hosted model, a deployed segmentation endpoint, or a trainable project model. Give it a name you can use in chat. Explicit model selection applies to that request; **Make default** changes future annotation. Training never changes the annotation default automatically.

- **VISTA3D:** local CT annotation and fine-tuning from immutable base weights.
- **U-Net:** 2D RGB or 3D scalar training from scratch, fine-tuning or continuation.
- **SAM 2.1 / MedSAM2:** local inference with editable boxes and points.
- **GPT-6 Astra:** default hosted annotation for pathology and video; radiology defaults to VISTA3D. Claude and Gemini can be named explicitly.

Manage project models through **Training → Rename / Delete**. Annotation versions link to the same model setup. Deleting a setup removes its versions from both tabs while preserving completed run records. Annotation-only connections are managed in Annotation; base models cannot be deleted. Resolve active jobs and default or dependent uses before deleting.

Save credentials in **Models → API keys** or use server environment references. Never put keys in chat. Back up `workspace/secrets.key` with the database. The [conversation model](coordinator.md) is configured separately from annotation. Provider contracts and configuration examples are in [providers](providers.md).

## Review regions and frame ranges

QuPath submissions can cover one or more selected regions of an imported image. Each region has its own revision and decision in **Reviews → Inspect**. CVAT submissions create review items for affected frame ranges; the inspection dialog lets reviewers step through source frames with or without the overlay. Accept an item or request changes independently. Correct in the native viewer, then resubmit: unaffected items keep their decisions. Overlapping pathology regions are rejected; reopen the existing region to revise it.

Accepted region footprints and video polygon ranges can train a 2D RGB U-Net. Pixels outside a reviewed region are excluded from loss, rather than labeled as background. Video samples retain source frame numbers and presentation timestamps. Rectangle tracks alone do not provide segmentation masks. Interpolated polygons whose vertex counts change need explicit keyframes before segmentation training; overlapping different classes must be corrected first.

## Train a model

In **Models → Training → Start training**, choose the structures and whether to evaluate:

- **Without evaluation:** the default for a new model with no evaluation choice. Train from accepted samples, including one source file, without requiring a separate reference dataset. No held-out score is reported.
- **Fixed:** a named independently labeled image/volume set excluded from all training.
- **Percentage-based:** a stable split belonging to this model. Choose a ratio such as 80:20 train/evaluation. New accepted cases extend it; another model may use a different ratio.

Existing assignments remain fixed. Patient, slide and procedure groups and identical sources stay together. Regions from one slide and frames from one video cannot be split between training and evaluation. A percentage split needs at least two independent source groups; one annotated video can train without evaluation. Previously trained cases cannot move into evaluation. Each run freezes its exact image and annotation revisions; later edits cannot change that run.

**Filter samples** is collapsed by default. It contains **Annotation status**, **Images**, then **Sample limit**. The image picker is a searchable table with ten rows per page and persistent selections. Filters affect training only; they never shrink saved evaluation references. Related cases stay together, so a sample limit may yield fewer cases than requested.

**Training settings (recommended)** overrides recommendations for this run. Epochs × training steps per epoch determines update count; patch sampling means an epoch need not visit every image. Batch size controls patches per update. VISTA3D averages accumulated gradients; U-Net batches patches together. Other settings include crop size, learning rate, device, seed and applicable spacing/window values. Structure mapping connects project labels to model classes.

## Compare and manage evaluation sets

Choose **Models → Evaluation → Compare models**. Both models use the same fixed, accepted reference version. A named set is prepared automatically when its labels are ready. An older saved version can be selected explicitly.

**Manage evaluation sets** supports search, rename, archive, restore and viewing cases. To add existing cases, select them in Datasets first. Archive hides a set while preserving reservations. Sets with saved references cannot be deleted. Unused evaluation images can be deleted before they enter saved references or training history; deleting an unused set alone does not release its reservations.

**Save evaluation references** publishes corrected or expanded references. Pending labels and incomplete structure coverage block publication. Known training lineage and duplicate images are excluded. Public data may overlap unknown upstream pretraining; independent project reservations cannot establish that it was unseen by the base model.

## Results and logs

Open **Activity → View logs** for training, evaluation and batch annotation. The text box shows the latest 1,000 lines, with 100/500-line choices and **Follow latest**. **Download full log** exports recorded job events. Settings appear above logs.

Runs without evaluation show training loss and the saved model, with no held-out score. **Results** for evaluated runs shows per-structure Dice/IoU, mean Dice, case count and reference version; training reports also include available losses. **Download report** exports JSON. A failed evaluation keeps the completed checkpoint; **Retry evaluation** retries scoring alone. Cancelled training does not publish an incomplete checkpoint.

Dice and IoU pool voxel counts across cases for each foreground structure. Mean Dice averages structure scores and excludes background; it is not a per-patient average. Empty, unreviewed or known overlapping references produce an error rather than a fabricated score. A fine-tuned model is not assumed to outperform its base.

## Try the Spleen learning workflow

Send these prompts one at a time in the main window. Wait for each job to finish; Activity shows progress and logs. Inspect the imported references and generated annotations before accepting their reviews.

<!-- spleen-prompts:start -->

> Create new project "Radiology - Spleen Segmentation"

> From Medical Decathlon Spleen Dataset import 80% images only for
> annotation and remaining 20% images+labels for evaluation.

> Mark all reviews as good for imported evaluation samples.

> Run VISTA3D segmentation to annotate spleen for first 5 images and submit
> for review.

> Mark all pending reviews as good

> Create VISTA 3D based model "VISTA3D-Spleen" for spleen

> Finetune VISTA3D-Spleen and use fixed Decathlon Spleen set for evaluation

> Compare VISTA3D-Spleen vs VISTA3D against fixed Decathlon Spleen set

<!-- spleen-prompts:end -->

The split imports 32 images for annotation and 9 image/label pairs for independent evaluation. Training creates a derived model; comparison reports Dice scores against the same held-out references.
