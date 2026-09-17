---
name: monailabel-model-evaluation
description: Compare trained and base models on fixed held-out references, manage evaluation sets, inspect scores, or explicitly promote a passing model.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: evaluate_candidate promote_candidate create_evaluation_set update_evaluation_set extend_evaluation_set publish_evaluation_set delete_evaluation_set
metadata:
  monailabel-context: project
---

# Model evaluation

Train normally on complete accepted annotations. Partial slices/regions and unreviewed predictions are not complete references; never bypass review on your own. A patient/slide source group belongs to one split. evaluate_candidate compares candidate and baseline on the SAME independently reviewed held-out reference version/snapshot. Use explicit candidate_name/baseline_name or current selected IDs; inspect to resolve missing selections. inspect_workspace(evaluations) returns measured scores. Newer does not mean better. Training/comparison never automatically changes defaults; promotion requires an explicit request and recorded passing criteria.

“Compare Organ model vs VISTA3D against the fixed set” → evaluate_candidate(candidate_name=Organ model, baseline_name=VISTA3D, evaluation_set_name=the exact available set name). Use names from workspace data; the service resolves current versions. Omit snapshot_id and evaluation_version_id when specifying a set.

Evaluation references must be accepted and independent of both models’ training. Imported evaluation annotations may still need review. Do not automatically approve them. Manage set membership or publish versions only when requested; ordinary comparison prepares eligible references through the evaluation service.
