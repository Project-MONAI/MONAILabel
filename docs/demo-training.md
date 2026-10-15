# Training in the bounded demo

The shared demo separates live annotation from recorded training. Set `MONAILABEL_DEMO_TRAINING_PREVIEW=1` on the demo server to disable new training jobs for every account, including administrators. The ordinary installation is unchanged when this flag is absent. This is a training-only resource restriction, not a public hosting or tenant-isolation boundary.

## Visitor experience

In **Models → Training**, **Recorded examples** shows two inline cards: **nnU-Net training** and **VISTA3D fine-tuning**. Their summaries load automatically; each card expands its recorded log and run details in place, without a popup. Examples are fetched independently and cached briefly in the browser; an unavailable example offers a retry without hiding the other card. **Preview training** on a training setup still opens an explicitly non-running walkthrough. It explains reviewed annotations, immutable snapshots, preprocessing, training, separate model versions and held-out evaluation. The walkthrough does not validate eligibility, save a snapshot or use the visitor's corrections.

Training requests through the assistant return the preview notice, without a job ID. Direct training endpoints reject the request with `403` and `demo_training_preview`. Both learning entry points reject work before creating snapshots, and the job scheduler independently rejects `train` submissions. A UI message alone is not the enforcement boundary. Creating a training setup remains possible; that is not a trained model.

Recorded examples show actual completion timestamps, run settings, case counts, elapsed time, checkpoint hashes, recorded logs and available scores. They never appear as newly running or completed visitor jobs. Missing or invalid examples fail visibly; the application does not generate placeholder scores or logs. Public data only. The prepared checkpoints and their predictions do not incorporate a visitor's subsequent corrections.

## Prepare real examples offline

Use a separate operator preparation process, with the GPU capacity reserved for it and no active visitor compute jobs. Reuse the existing environment and cache paths. The script requires verified cached Medical Decathlon Spleen data and the pinned VISTA3D checkpoint; a cache miss stops preparation. It does not provision models, viewers or a chat service.

```bash
uv run python scripts/demo_prepare_training.py \
  --output /path/to/new-offline-examples \
  --source-revision <exact-40-character-source-commit>
```

The output path must not already exist. A dedicated workspace retains three training cases, one separate evaluation case, real model checkpoints, immutable snapshots and job logs. Dataset-provided masks are explicitly identified as operator preparation references, not clinician corrections. nnU-Net uses a bounded 20 × 32-update run by default; VISTA3D uses 5 × 10 updates. These budgets demonstrate the workflow and do not promise improved accuracy. The recorded duration includes preparation, checkpoint writing and requested evaluation. A saved checkpoint can survive an evaluation failure; its example then reports no score and explains the failure.

The script explicitly enables training only in its own offline process. It must not be used as a web-accessible command. Apply external memory, CPU and runtime limits to that process. Never point a second server at the live workspace. Neither the private preparation workspace nor its authentication database belongs in the public catalog.

Set `MONAILABEL_DEMO_TRAINING_EXAMPLES=/path/to/new-offline-examples/catalog` on the demo server after checking the exported JSON files. The authenticated, read-only `/api/demo-access/training/{recipe}` endpoint accepts only `nnunet-v2` and `vista3d`. It serves a bounded, schema-checked evidence document, not arbitrary files, credentials or executable model uploads. Keep this catalog operator-owned and unwritable by viewers/visitors.

## Scope and checks

The prepared workspace retains the actual models for subsequent inspection. The catalog currently presents evidence only; it does not import a checkpoint or its predictions into a visitor's project. Sharing models with viewer-ready private case copies is a separate integration step. Live evaluation and inference remain separate GPU workloads and still require the demo's admission and resource controls. Self-registration, tenant isolation and automatic account cleanup are not provided by the training preview.

Tests cover direct and assistant-triggered training, no new snapshots/jobs, authenticated read-only evidence, malformed/oversized records, path traversal and symlink rejection, missing examples and browser escaping. A real-browser test verifies that opening the preview sends no training POST and creates no job. Run these checks in disposable workspaces; never mutate a user's current project.
