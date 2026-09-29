# Datasets, models and learning

## Import data

In **Datasets → Import files**, choose **Annotation & training** or **Evaluation only**. Evaluation-only groups stay out of every model's training.

Image/label pairs match by filename; label suffixes may be `_mask`, `_label`, `_labels`, `_seg` or `_segmentation`. Map foreground values under **Structures in these labels**; zero is background. Evaluation imports accept the supplied references immediately and exclude them from training. For annotation/training imports, mark labels reviewed only after inspecting them.

Use **Advanced options** for patient/slide grouping and provenance. Related cases and exact decoded-image duplicates stay together. Re-imports preserve existing annotations; partial failures identify cases to retry.

Supported images are scalar NIfTI and bounded PNG/JPEG/TIFF/SVS. Labels must be integer NIfTI/PNG/TIFF maps with matching geometry. Limits: 128 MiB per file, 256 MiB decompressed NIfTI, 67,108,864 voxels, 16,777,216 image pixels and 31 foreground structures. See [DICOM import](viewers.md#dicom-import-and-viewing) and [video import](video.md).

## Sample datasets

Use **Datasets → Sample datasets** for [downloadable templates](datasets.md). Choose images only, images with labels or unlabeled test images where available. Evaluation imports require labels.

```text
Import 10 spleen images from Medical Decathlon to start annotation

Import 80% of Decathlon spleen images for annotation and 20% with labels for evaluation
```

A combined percentage import uses the whole labeled section unless a count is given and rounds evaluation up to whole source groups. It is separate from each model's training split. Published evaluation references are accepted on import; labels in the annotation/training portion still need review.

## Review annotations

Start dataset-wide batch annotation from the main workspace. Viewer prompts annotate or edit the current image, volume or video. Exact “Undo”, “Undo that”, “Redo” and “Redo that” commands use the current viewer’s edit history without a model call.

Inspect and correct annotations in the viewer, then submit. In **Reviews**, select rows and choose **Review decision → Good / Needs changes / Pending**. Decisions apply to submitted revisions; selections persist across pages.

## Create and manage models

**Models → Add model** connects a hosted/deployed endpoint or creates a trainable model. Use its name in chat. **Make default** changes future annotation; training does not change the default.

- VISTA3D: CT annotation and fine-tuning.
- TotalSegmentator CT / MRI: separate 3 mm models, annotation, fine-tuning and continuation.
- U-Net: 2D/3D scratch training, fine-tuning and continuation.
- nnU-Net v2: automatically planned CT / single-sequence MRI training for custom targets; see [setup](providers.md#nnu-net-v2).
- SAM 2.1 / MedSAM2: local inference with spatial prompts.
- Hosted vision: image/slice annotation, localization and classification through supported adapters.

Use **Training → Rename / Delete** for trainable models and **Annotation** for annotation-only connections. Base models cannot be deleted. Active jobs and dependent uses must be resolved first. Completed run records are retained.

Store keys in **Models → API keys** or server environment variables. Back up `secrets.key` with the database. See [model connections](providers.md).

## Review regions and frame ranges

**Reviews → Inspect** supports separate decisions for QuPath regions and CVAT frame ranges. Correct in the viewer and resubmit; unaffected scopes keep their decisions. Revise an existing pathology region instead of adding an overlapping region.

Accepted regions and polygon frame ranges can train a 2D U-Net. Unreviewed pixels are excluded from loss. Boxes alone cannot train segmentation. Polygons with changing vertex counts need explicit keyframes; conflicting class overlaps must be corrected.

## Train a model

In **Models → Training → Start training**, choose structures and an evaluation option:

| Option | Behavior |
| --- | --- |
| Without evaluation | Train accepted samples, including a single source; report no held-out score |
| Fixed | Use a named accepted reference set excluded from all training |
| Percentage-based | Maintain a stable, model-specific split over independent source groups |

Patient, slide and procedure groups stay together. Percentage splits need at least two groups. Existing assignments remain fixed; previously trained cases cannot move into evaluation. Each run freezes image and annotation revisions.

**Filter samples** narrows training only; evaluation references stay fixed. Grouping can make a sample limit return fewer cases. **Training settings** overrides epochs, steps, batch/crop size, learning rate, device and other recipe settings for the run. Patch-sampled epochs need not visit every image.

## Compare and manage evaluation sets

Use **Models → Evaluation → Compare models** to score both models against the same accepted reference version. **Save evaluation references** publishes corrected or expanded labels; pending or incomplete coverage blocks publication.

**Manage evaluation sets** supports rename, archive, restore and case inspection. Select existing cases in Datasets before adding them. Archiving or deleting a set does not release evaluation reservations; sets with saved references cannot be deleted.

Known training lineage and duplicate images are excluded. Project reservations cannot establish whether a base model saw public data during upstream pretraining.

## Results and logs

**Activity → View logs** shows job progress; **Download full log** exports events. **Results** shows losses, model/checkpoint information and available evaluation scores; **Download report** exports JSON.

Dice/IoU pool voxel counts per foreground structure. Mean Dice averages those structure scores. Runs without evaluation have no held-out score. A failed evaluation keeps the completed checkpoint; **Retry evaluation** retries scoring. Cancelled training publishes no incomplete checkpoint.

## Try the Spleen learning workflow

Send each prompt separately and wait for completion. Inspect labels before accepting reviews.

<!-- spleen-prompts:start -->

> Create new project "Radiology - Spleen Segmentation"

> From Medical Decathlon Spleen Dataset import 80% images only for annotation and remaining 20% images+labels for evaluation.

> Run VISTA3D segmentation to annotate spleen for first 5 images and submit for review.

> Mark all pending reviews as good

> Create VISTA 3D based model "VISTA3D-Spleen" for spleen

> Finetune VISTA3D-Spleen and use fixed Decathlon Spleen set for evaluation

> Compare VISTA3D-Spleen vs VISTA3D against fixed Decathlon Spleen set

<!-- spleen-prompts:end -->
