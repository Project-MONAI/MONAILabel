---
name: monailabel-model-training
description: Create models from training recipes, create snapshots, train, fine-tune or continue learning. Annotating with an existing trained model uses the source's annotation skill instead.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: create_learner start_training create_snapshot
metadata:
  monailabel-context: project
---

# Model training

Using an existing model to annotate or segment is inference. Load monailabel-radiology for a volume/slice, monailabel-pathology for an image/region, or monailabel-video for a video/frame, then call its annotation tool. Training requires an explicit request to train, fine-tune, continue training or create a training setup; a selected learner does not request training.

For questions about default training parameters, inspect_workspace(collection=training_recipes, recipe_id=the requested recipe). These are the actual hyperparameters; action-tool arguments are not training defaults. Use learners to inspect a named setup's saved config.

A learner is a training setup; a model is an inference configuration/checkpoint, with different IDs. create_learner preserves the exact requested name. recipe=monai-unet supports scratch (no parent) for 2D RGB or 3D scalar data. recipe=vista3d, recipe=totalsegmentator-ct, and recipe=totalsegmentator-mr with initialization=fine_tune derive from the matching read-only base. TotalSegmentator CT and MRI use different checkpoints: retain the modality the user requests. If architecture/parent is missing, ask. Set start_now=true only for explicit train/finetune now; merely creating a model uses false. VISTA3D and TotalSegmentator may omit targets to inherit the base vocabulary; explicit organs set initial training choices. U-Net and nnU-Net require fixed output classes. recipe=nnunet-v2 trains single-channel CT/MRI volumes from scratch; require modality=CT or modality=MRI from the user or known dataset, never guess from filenames. Use it for custom structures such as liver tumors or MRI prostate zones. Planning uses only training cases; the selected held-out set is scored after training. Continuation retains the preprocessing plan and optimizer; fine-tuning an existing project checkpoint retains its plan with a new optimizer. Both require unchanged modality and target mapping. nnU-Net uses the ResEnc L preset with a short default run of 20 epochs × 32 updates. Pass explicitly requested epochs and steps per epoch in start_training.config; do not replace them with defaults or silently limit samples. A short run does not establish model quality; inspect held-out results and predictions.

start_training uses learner_name for an explicitly named existing setup, overriding context; do not create a duplicate. Its saved setup already determines the architecture, modality and output classes; do not ask the user to specify these again. It freezes a NEW snapshot of currently eligible annotations, including newly reviewed data. VISTA3D supports per-run targets; continuing optimizer state requires unchanged training organs. Continuing a checkpoint uses mode=continue and its id, never its parent_id. Omit parent_model_id to use the selected checkpoint. Check current recent_jobs rather than stale history before assuming training is still running.

A model can train without evaluation, use a fixed evaluation_set_id, or reserve its own validation_percentage (80:20 means 20). Omit both to retain a saved choice; when none exists, train approved samples without requiring a separate evaluation dataset. Set validation_percentage=0 for an explicit request without evaluation. Never invent an evaluation score. Pass an explicit ratio to create_learner when start_now=true, or to start_training for an existing setup. Percentage sets grow with eligible annotations, preserving prior assignments and snapshots; never reuse another model's percentage set. Fixed evaluation-only sets are excluded from every model's training. Do not carry comparison context into training unless requested. Inspect evaluation_sets/evaluation_set_versions to resolve fixed references. Create, extend, publish or archive only on request.

Train on accepted full annotations or accepted scoped coverage. Unreviewed pixels are excluded from loss; boxes are not segmentation masks. Keep all regions from a slide and all frames from a patient/procedure together in one model split. Evaluation-only cases stay out of training. Do not substitute snapshot creation for a training request.

Examples:

- “Create CT nnU-Net v2 model Liver specialist for liver and tumor” → create_learner(recipe=nnunet-v2, modality=CT, name=Liver specialist, targets=[liver, tumor]). Use imported label names exactly; do not add synonymous labels.

- “Train Lung nnU-Net for 1000 epochs, 250 steps each; use the fixed evaluation set” → start_training(learner_name=Lung nnU-Net, config={epochs: 1000, steps_per_epoch: 250}, evaluation_set_name=the exact available set name). The MRI prostate example requests 500 epochs instead.

- “Create a VISTA3D spleen model named Organ model” → create_learner(recipe=vista3d, name=Organ model, targets=[Spleen]); omit start_now.
- “Finetune Organ model using the fixed evaluation set” → start_training(learner_name=Organ model, evaluation_set_name=the exact available set name). Omit targets to use its configured structures.
- “Continue training the selected model” → start_training(mode=continue). This resumes its optimizer state; fine_tune starts a new optimizer and is a different request.
- “Fine-tune Organ model with approved samples” → start_training(learner_name=Organ model). Do not demand an evaluation dataset.
- “Fine-tune Organ model with an 80:20 train/evaluation ratio” → start_training(learner_name=Organ model, validation_percentage=20).
- “Train Organ model with 25% validation” → start_training(learner_name=Organ model, validation_percentage=25).

Use the smallest arguments needed. targets contains foreground names only; Background is included automatically. “All imported labels” means every foreground name in the project. Label IDs in workspace data are identifiers, not target names. Omitted settings use recommended or saved defaults.
