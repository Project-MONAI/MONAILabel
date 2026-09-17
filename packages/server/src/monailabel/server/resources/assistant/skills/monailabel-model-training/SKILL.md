---
name: monailabel-model-training
description: Create a named trainable model or a reviewed dataset snapshot; train, fine-tune, or continue an existing model with fixed or percentage validation.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: create_learner start_training create_snapshot
metadata:
  monailabel-context: project
---

# Model training

A learner is a training setup; a model is an inference configuration/checkpoint, with different IDs. create_learner preserves the exact requested name. recipe=monai-unet supports scratch (no parent) for 2D RGB or 3D scalar data. recipe=vista3d, initialization=fine_tune derives from its read-only base. If architecture/parent is missing, ask. Set start_now=true only for explicit train/finetune now; merely creating a model uses false. VISTA3D may omit targets to inherit the base vocabulary; explicit organs set initial training choices. U-Net requires fixed output classes.

start_training uses learner_name for an explicitly named existing setup, overriding context; do not create a duplicate. It freezes a NEW snapshot of currently eligible annotations, including newly reviewed data. VISTA3D supports per-run targets; continuing optimizer state requires unchanged training organs. Continuing a checkpoint uses mode=continue and its id, never its parent_id. Omit parent_model_id to use the selected checkpoint. Check current recent_jobs rather than stale history before assuming training is still running.

Each model uses an existing fixed evaluation_set_id OR its own validation_percentage (80:20 means 20). Omit both to retain the saved choice. Percentage sets grow with eligible annotations, preserving prior assignments and snapshots; never reuse another model's percentage set. Fixed evaluation-only sets are excluded from every model's training. Do not carry comparison context into training unless requested. Inspect evaluation_sets/evaluation_set_versions to resolve fixed references. Create, extend, publish or archive only on request.

Train on accepted complete references; source groups and evaluation-only cases stay out of training. Do not substitute snapshot creation for a training request.

Examples:

- “Create a VISTA3D spleen model named Organ model” → create_learner(recipe=vista3d, name=Organ model, targets=[Spleen]); omit start_now.
- “Finetune Organ model using the fixed evaluation set” → start_training(learner_name=Organ model, evaluation_set_name=the exact available set name). Omit targets to use its configured structures.
- “Continue training the selected model” → start_training(mode=continue). This resumes its optimizer state; fine_tune starts a new optimizer and is a different request.
- “Train Organ model with 25% validation” → start_training(learner_name=Organ model, validation_percentage=25).

Use the smallest arguments needed. Label IDs in workspace data are identifiers, not target names. Omitted settings use recommended or saved defaults.
