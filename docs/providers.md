# Provider and training contracts

## Bring your own vision model

In **Models → Add model → Use a hosted vision model**, choose a service, load its compatible models, select one and give it a project name.

| Service | Adapter | Server key variable |
| --- | --- | --- |
| OpenAI | `openai-polygons` (Responses) | `OPENAI_API_KEY` |
| Anthropic | `anthropic-polygons` (Messages) | `ANTHROPIC_API_KEY` |
| Gemini | `openai-chat-polygons` | `GEMINI_API_KEY` |
| NVIDIA | Responses or Chat Completions, from catalog mode | `NV_INFERENCE_API_KEY` |

Alternatively, save an encrypted project key in the import form. Rotate it under **Models → API keys**. Imported models retain their endpoint and credential reference. Names must be unique within the project; duplicate connections are marked **Already added**.

Discovery requires image input and structured annotation output. NVIDIA shows the two newest versions per family, keeping hosting routes separate. For other services, use **Connect another vision API** with an explicit endpoint and model ID. Annotation models are independent of the [conversation coordinator](coordinator.md).

### Hosted presets

**Models → Annotation** groups predefined models into **Radiology segmentation** (VISTA3D and TotalSegmentator CT/MRI), **Interactive segmentation** (nnInteractive, SAM 2.1 and MedSAM2), and **Vision-language models**. Trained project models and manually added connections have their own sections.

VISTA3D weights and derivatives are restricted to noncommercial research/evaluation. MedSAM2 weights are restricted to research/education. nnInteractive weights use CC BY-NC-SA 4.0 (noncommercial, attribution and share-alike). See [model licenses and citations](../THIRD_PARTY_NOTICES.md#downloaded-model-weights).

GPT-6 Astra, Claude Opus 5 and Gemini 3.8 Flash use fixed model identities. Automatic connection selection prefers NVIDIA, then the corresponding direct provider. Unavailable presets are hidden.

Use **Change provider** on a preset to select another available connection. The choice persists for that project and requires an idle project. **Automatic · NVIDIA first** restores automatic selection. Imported connections have fixed providers. Inference errors do not switch accounts or models.

New projects use GPT-6 Astra as the annotation default when its connection is available; otherwise they start with local VISTA3D. Naming a model in a prompt overrides the default. Existing project and structure choices are preserved. See [manual connection examples](../examples/models).

## Credentials and configuration

Use `token_env` or `credential_id`, never a raw key. Credentials are resolved per invocation, so key rotation requires no restart.

```json
{
  "name": "My segmentation model",
  "provider": "http-mask",
  "label_ids": [0, 1, 2],
  "config": {
    "url": "http://127.0.0.1:9000/predict",
    "model": "a-pinned-model-version",
    "token_env": "MY_MODEL_API_KEY",
    "timeout": 120
  }
}
```

## HTTP mask

The adapter sends a POST with an optional bearer token:

```json
{
  "image": [[[0.1], [0.8]], [[0.2], [0.9]]],
  "spatial_shape": [2, 2],
  "labels": [
    {"id": 0, "name": "Background", "color": "#000000"},
    {"id": 1, "name": "Target", "color": "#50a188"}
  ],
  "prompt": "Segment Target",
  "model": "a-pinned-model-version"
}
```

Expected response:

```json
{"mask": [[0, 1], [0, 1]]}
```

Images use `H×W×C` or `I×J×K×1` arrays. RGB values use `[0,1]`; NIfTI retains source scaling. Return an integer mask on the same spatial grid using registered class IDs. The endpoint owns normalization, weights and accelerator setup. Service-specific NIM APIs may need a wrapper.

## Hugging Face

Connect a deployed [image-segmentation endpoint](https://huggingface.co/docs/inference-providers/tasks/image-segmentation). The adapter sends PNG bytes and expects `{label, mask, score}` items with base64 binary PNG masks. Configure `label_map` from returned names to project IDs. Unknown labels, changed dimensions and overlapping classes are rejected. A Hub repository ID alone does not install a runtime.

## Vision API adapters

| Adapter | Request | Output budget |
| --- | --- | --- |
| OpenAI Responses | Image input and strict polygon JSON schema | `max_output_tokens` |
| OpenAI-compatible Chat Completions | Image messages and `response_format.json_schema` at a full `/chat/completions` URL | `max_completion_tokens`, or configured `max_tokens_field` |
| Anthropic Messages | PNG image blocks and `output_config.format` at `/v1/messages` | `max_tokens` |

`max_output_tokens` in model configuration maps to the field above. `reasoning_effort` is supported only by compatible adapters; Anthropic rejects GPT-specific settings. Gemini uses `max_tokens` at `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`.

Polygon coordinates use original 2D image pixels. Refused, incomplete, malformed, out-of-bounds or conflicting output fails validation without applying a partial result.

Volume requests with 2D models require an explicit slice and intensity window. `AnnotateRequest.all_slices=true` iterates the selected source axis and publishes only after every plane succeeds. Lossless `orientation` transforms support transpose and row/column flips; rotations requiring resampling are rejected. Results are restored to the source grid before merging.

References: [OpenAI image input](https://developers.openai.com/api/docs/guides/images-vision), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai).

## TotalSegmentator CT and MRI

Both models are available under **Models**: CT supports 117 structures and MRI supports 50. Select the model matching the scan modality. They use the upstream 3 mm checkpoints; predictions return on the original image grid. Weights download on first use and are checksum-verified under `MONAILABEL_MODELS_DIR` (the workspace model cache by default). Inference stays local and upstream usage telemetry is disabled.

Choose **Create project model**, or use chat after accepting annotation reviews:

```text
Fine-tune TotalSegmentator CT for spleen and liver as "Abdominal CT".

Fine-tune TotalSegmentator MRI for prostate as "Prostate MRI".

Continue training Prostate MRI.
```

Fine-tuning uses the original nnU-Net architecture and preprocessing, with the reviewed structures defining the loss. Unreviewed voxels are excluded. Training saves new weights and optimizer state with parent lineage; base weights remain unchanged. Epochs, steps, batch size and learning rate are optional run settings. Scratch training and changing the base resolution are unavailable for these recipes.

Use [sample datasets](datasets.md#try-totalsegmentator) for annotation and training practice. Public TotalSegmentator collections include upstream development data and cannot evaluate TotalSegmentator or its descendants as independent references.

The supported tasks and checkpoints use Apache-2.0. Cite the [CT paper](https://doi.org/10.1148/ryai.230024), [MRI paper](https://doi.org/10.1148/radiol.241613) for MRI, and [nnU-Net](https://doi.org/10.1038/s41592-020-01008-z). See [third-party notices](../THIRD_PARTY_NOTICES.md).

## nnU-Net v2

Choose **Add a model → Train a project model → nnU-Net v2**, select CT or MRI, and choose your targets. Use single-channel 3D scans with consistent acquisition type; MRI models should use one sequence. Useful targets include CT liver tumors and MRI prostate zones with corresponding reviewed masks.

nnU-Net plans spacing, normalization, patch and batch sizes, and the network from training cases using the upstream [ResEnc L preset](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/documentation/resenc_presets.md). MONAI handles planning/preprocessing through `nnUNetV2Runner`; nnU-Net trains one `3d_fullres` configuration. Validation and independent evaluation cases remain outside planning and optimization. MONAI Label scores the final checkpoint on the selected held-out references. No cross-validation ensemble or automatic best-checkpoint selection is performed.

```text
Create CT nnU-Net v2 model "Liver specialist" for liver and tumor.

Train Liver specialist on approved masks; use the fixed evaluation set.

Continue training Liver specialist.
```

The default is a short run of 20 epochs × 32 updates. Training uses SGD/Nesterov and a polynomial learning-rate schedule starting at 0.01. Request longer runs in chat or **Start training → Training settings**. The README uses 1,000 × 250 for CT lung tumors and 500 × 250 for MRI prostate zones, with all labeled cases and 20% held out. Assess per-class Dice/IoU and inspect predictions before using a trained model.

CT lung tumor models can be compared with VISTA3D using its explicit [lung tumor class (23)](https://huggingface.co/MONAI/vista3d/blob/c6dbe159632a4767696e09f91d74d729b82e73e6/docs/labels.json). VISTA3D is a CT model and does not provide MRI prostate-zone classes. Its upstream training sources include Decathlon Lung; use independently sourced cases for an independent benchmark.

Training preserves geometry and excludes unreviewed voxels from loss. Checkpoints retain the preprocessing plan, class mapping, normalization, weights and optimizer. Continue keeps the parent's plan and optimizer; fine-tuning a project checkpoint keeps its plan and weights with a fresh optimizer. Both publish a new version. Modality and target mapping must match the parent.

Multi-sequence MRI, arbitrary upstream checkpoint imports and ensembles are not supported. See the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet) and [license and citation](../THIRD_PARTY_NOTICES.md#nnu-net-v2).

## Adding a provider or training recipe

1. Implement `Segmenter.predict` or `Trainer.train` from `core/ports.py`.
2. Validate configuration, dimensions, geometry and labels.
3. Register the provider in `server/models/service.py` or the recipe in `server/recipes.py`.
4. Record model state and training lineage; keep weights in artifacts or the runtime cache.
5. Test output validation, cancellation and held-out evaluation.

For catalog discovery, add verified IDs to `providers/catalog/compatibility.py` after checking image input, structured output and transport support.

Training recipes are 2D/3D `monai-unet`, `nnunet-v2`, `vista3d`, `totalsegmentator-ct` / `totalsegmentator-mr` fine-tuning, and `pixel-gaussian`. The Gaussian baseline continues with previously unseen accepted cases; fine-tuning halves parent statistics before adding cases. Changed labels on previously trained cases require scratch retraining.
