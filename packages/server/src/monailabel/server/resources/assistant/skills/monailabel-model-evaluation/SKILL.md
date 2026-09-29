---
name: monailabel-model-evaluation
description: Explicitly evaluate or compare models on held-out references, inspect scores, manage evaluation sets, or promote a model. Annotating a source is not evaluation.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: evaluate_candidate promote_candidate create_evaluation_set update_evaluation_set extend_evaluation_set publish_evaluation_set delete_evaluation_set
metadata:
  monailabel-context: project
---

# Model evaluation

“Segment using a trained model” is annotation, not evaluation. Load the current source's radiology, pathology or video annotation skill and call its annotation tool. A selected evaluation set or previous comparison does not change this request into evaluation.

Train on accepted annotations or accepted region/frame coverage; exclude unreviewed pixels. A patient/slide/procedure source group belongs to one split. evaluate_candidate starts a NEW comparison of candidate and baseline on the SAME independently reviewed held-out reference version/snapshot. Use explicit candidate_name/baseline_name or current selected IDs; inspect to resolve missing selections. Newer does not mean better. Training/comparison never automatically changes defaults; promotion requires an explicit request and recorded passing criteria.

“Compare the selected model against the selected previous version on the current held-out set” → evaluate_candidate() with no arguments when context already contains model_id, baseline_id and an evaluation set/version or snapshot. Omitted arguments use those exact selections; do not copy IDs from older turns or replace a selected checkpoint with its parent. Supply names or IDs only to change a selection explicitly requested by the user.

“Show the held-out evaluation results” → inspect_workspace(collection=evaluations). Showing, listing or explaining existing results is read-only. Never start evaluate_candidate for a request to see scores, even when the conversation previously ran a comparison. If no results exist, explain that; wait for an explicit request to run evaluation.

“Compare Organ model vs VISTA3D against the fixed set” → evaluate_candidate(candidate_name=Organ model, baseline_name=VISTA3D, evaluation_set_name=the exact available set name). Use names from workspace data; the service resolves current versions. Omit snapshot_id and evaluation_version_id when specifying a set.

Evaluation references must be accepted and independent of both models’ training. Published labels imported for evaluation are accepted by the import service; user-created references need review. Never change review decisions merely to enable evaluation. Manage set membership or publish versions only when requested; ordinary comparison prepares eligible references through the evaluation service. A split excludes project training overlap but cannot establish independence from an upstream pretrained model's development data.
