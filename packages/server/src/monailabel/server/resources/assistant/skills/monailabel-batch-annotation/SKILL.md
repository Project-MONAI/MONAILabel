---
name: monailabel-batch-annotation
description: Segment several separate dataset images in one batch job, optionally submitting predictions for review. Use for requests with an image count; one volume, slice or region belongs to radiology or pathology.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: annotate_batch
metadata:
  monailabel-context: project
---

# Batch annotation

Use annotate_batch for segmentation of several existing images. Resolve the exact available model_name; omit it only when the user wants the current default. targets contains structure names, never numeric IDs. Honor the requested count through limit.

“Annotate spleen on the first five images using VISTA3D and submit them” → model_name=VISTA3D, targets=[Spleen], limit=5, submit_for_review=true. Submission creates pending reviews, never acceptance.

The service skips evaluation-only images, saved annotations and pending proposals. The reply starts a job; Activity reports progress and logs. Selecting cases alone does not segment them. Do not change the default annotation model to satisfy a one-time request.
