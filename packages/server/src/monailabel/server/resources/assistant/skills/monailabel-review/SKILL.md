---
name: monailabel-review
description: Accept saved annotations as good, request changes, or reset reviews to pending; submit or review the current viewer draft.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: review_saved_annotations review_annotation viewer_edit
metadata:
  monailabel-context: project
---

# Review

Submission is separate from acceptance. annotate_batch(submit_for_review=true) generates masks and saves pending revisions. viewer_edit(submit) saves the current complete viewer draft as pending. Never accept merely because the user requested submission.

Workspace decisions use review_saved_annotations: Good=accepted, Needs changes=changes_requested, Reset to pending=pending. “All pending reviews as good” means scope=pending, verdict=accepted, dataset_use=all. Restrict dataset_use to evaluation or annotation only when the CURRENT request explicitly does so; never carry an earlier filter forward. This acts on saved annotations without opening a viewer.

Inside a viewer, review_annotation records accepted (including corrections) or changes_requested. The viewer must save before claiming success. No decision is NOT rejection. Leaving pending/skipping needs no modifying tool. Stale revisions require reload; never accept a superseded annotation. Reviewers can correct and accept without model-management permission.
