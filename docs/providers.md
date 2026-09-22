# Provider and training contracts

Provider configurations store credential references, never raw API keys:

```json
{
  "name": "My multi-structure model",
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

Alternatively use `credential_id` from the credential API/UI; choose only one credential mechanism. Credentials are resolved on each invocation, so rotation does not require restarting the server. Plaintext keys are not returned by model/credential listing or validation errors.

### Bring your own vision model

In **Models → Add model → Use a hosted vision model**, choose NVIDIA gateway, OpenAI, Anthropic (Claude) or Google (Gemini). If the server has that provider's environment key, its compatible model list loads automatically. Search by name or model ID, select a model, give it a name for prompts and choose **Add model**. Models using the same connection already in the project are marked **Already added**. Imported models belong only to this project and retain their provider and endpoint throughout their life. Their names can be changed; names must be unique within the project.

Discovery uses `NV_INFERENCE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` from the server environment. Alternatively, choose **Save a new API key** and **Load models** to store an encrypted key, or **Use a saved API key** to reuse one in the project. Each imported model retains that credential reference. Saved keys can be rotated under **Models → API keys**. A chat subscription is not an API credential.

Only models compatible with image input and structured annotation output appear. Discovery reads account-scoped metadata without running inference; import rechecks availability with the same key. Anthropic capability flags and NVIDIA capability metadata take precedence when provided. OpenAI and Gemini listings, and NVIDIA routes without capability metadata, use a conservative compatibility registry. Unknown models, unsupported variants and gateway routes outside Chat Completions or Responses are hidden. NVIDIA's model mode selects the matching API adapter and endpoint automatically. This verifies API compatibility, not annotation accuracy; review model output before submitting it. For a service outside this catalog, **Connect another vision API** retains manual model ID, endpoint and credential setup.

The registry is in `providers/catalog/compatibility.py`. Add exact IDs after checking image input, JSON-schema output and the configured API transport in the provider documentation; do not infer compatibility from a model name containing “vision.” Dated snapshots inherit a known model's compatibility, with exceptions for older incompatible releases. Sources: [OpenAI model capabilities](https://developers.openai.com/api/docs/models), [Anthropic model capabilities](https://platform.claude.com/docs/en/api/models/list), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Gemini models](https://ai.google.dev/api/models) and [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai).

The NVIDIA import picker keeps the two newest available versions per model family, including their size variants. GPT versions are ranked together; Claude Opus, Sonnet and Haiku and Gemini Pro, Flash and Flash Lite are separate families. Hosting routes such as Azure, AWS and Switchyard remain separate choices, identified in the picker and model card. This filter does not remove existing imports or change the three fixed presets. Direct-provider catalogs are not limited by this NVIDIA-specific recency filter.

Discovered models use these service adapters and fixed provider endpoints:

| Service | Annotation provider | API key environment variable |
| --- | --- | --- |
| OpenAI | `openai-polygons` (Responses) | `OPENAI_API_KEY` |
| Anthropic | `anthropic-polygons` (Messages) | `ANTHROPIC_API_KEY` |
| Gemini | `openai-chat-polygons` (Google's compatibility API) | `GEMINI_API_KEY` |
| NVIDIA gateway | `openai-chat-polygons` or `openai-polygons`, from catalog mode | `NV_INFERENCE_API_KEY` |

These models can segment images or selected volume slices, locate targets, classify existing objects and seed video tracking. SAM2 remains the temporal tracker. Annotation connections are separate from the [conversation coordinator](coordinator.md).

### HTTP mask

`POST` to the configured endpoint, with an optional bearer token:

```json
{
  "image": [
    [[0.1], [0.8]],
    [[0.2], [0.9]]
  ],
  "spatial_shape": [2, 2],
  "labels": [
    { "id": 0, "name": "Background", "color": "#000000" },
    { "id": 1, "name": "Target", "color": "#50a188" }
  ],
  "prompt": "Segment Target",
  "model": "a-pinned-model-version"
}
```

Expected response:

```json
{
  "mask": [
    [0, 1],
    [0, 1]
  ]
}
```

The image is feature-last: `H×W×C` for 2D, `I×J×K×1` for scalar NIfTI. NIfTI intensity values retain source scaling; RGB images use `[0,1]`. The response must be an integer mask with exactly the input spatial shape and registered class IDs. Model-specific normalization, inference, weights, and accelerator setup belong in the provider runtime. No fallback to another model occurs when a selected provider fails.

A NIM endpoint may require a wrapper translating its specific request/response format; it is not assumed to implement this generic mask contract.

### Hugging Face

The adapter sends PNG bytes to an explicitly deployed segmentation endpoint and expects `{label, mask, score}` items, with `mask` a base64 binary PNG. An explicit `label_map` maps returned names to project IDs. Unknown labels, changed dimensions, or conflicting class overlaps fail validation. Registering a Hub repository name alone does not install its runtime.

Contract reference: [Hugging Face image segmentation](https://huggingface.co/docs/inference-providers/tasks/image-segmentation).

### OpenAI polygons

The adapter sends image input and a strict polygon JSON schema to the configured Responses endpoint. Output coordinates refer to original 2D image pixels; polygons are rasterized and bounds/class overlaps checked. This provides editable proposals, not a claim of validated medical masks.

A full NIfTI volume is never silently converted to an image. A caller can explicitly select a source slice and intensity window; the backend then forms the 2D input and merges the result only into that slice. Fine-tuning is not inferred from API inference support.

For explicit whole-volume annotation with a 2D model, `AnnotateRequest.all_slices=true` requires the same slice plane/window/orientation context. The backend iterates every source index on that axis, checks cancellation between provider calls, and publishes one proposal only after every plane succeeds. Each plane is transformed for display before inference and restored to source coordinates afterwards. The proposal records `all_slices`, the `volume_plane` used for inference, and a null `slice` edit scope so older clients also merge the whole volume. Native volume providers keep their volume inference path.

Contract references: [image inputs](https://developers.openai.com/api/docs/guides/images-vision), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

### OpenAI-compatible Chat Completions

`openai-chat-polygons` shares the polygon schema/rasterizer with the Responses adapter. It sends image messages, `response_format.json_schema`, an explicit model identifier, a configurable completion budget, and optional reasoning effort to a full `/chat/completions` URL. Incomplete, refused, malformed, out-of-bounds, and overlapping responses fail validation. The gateway is not inferred from a model name and there is no model fallback.

The three hosted presets have fixed names and model identities: **GPT-6 Astra**, **Claude Opus 5** and **Gemini 3.5 Flash**. At server startup, read-only catalog checks discover matching NVIDIA routes and their API formats dynamically. Automatic selection uses NVIDIA's unqualified model alias when offered, otherwise the first matching compatible route returned by the catalog. If no NVIDIA route is available, the corresponding OpenAI, Anthropic or Gemini key is checked for the same model. Presets without an available connection are hidden; local VISTA3D/SAM presets remain available. These checks do not run inference.

Each card shows its current provider and NVIDIA hosting route. **Change provider** applies only to these hosted presets and selects another available connection to the same model. All matching NVIDIA routes are offered, so you can explicitly choose the hosting service whose cost you prefer. You can use a server environment key or a project key saved through the form. A manual selection is project-specific and survives restarts; it does not fall back if its provider becomes unavailable. Choose **Automatic · NVIDIA first** to restore automatic selection. Provider changes check the model version and require an idle project, preserve historical connection records and move project defaults to the replacement. Predefined names cannot be edited. Normal imports have no provider-switch control.

Select a model in your viewer or name it in chat, for example, `Segment the tool on this frame using Claude Opus 5`. For video tracking, the selected vision model creates the initial box or polygon; SAM2 propagates it. GPT-6 Astra is the default hosted annotation model for pathology and video; VISTA3D remains the radiology default. Compatible project/target defaults and dedicated target models take precedence. Claude and Gemini require an explicit choice or configured default. Retired Sol presets remain attached to historical annotations; their defaults move to Astra when available. Imported Sol connections are retained. Annotation failures never trigger a retry through a different account. The manual registration examples in [`examples/models`](../examples/models) illustrate fixed project connections; discover current model routes before using them.

Optional `max_output_tokens` maps to `max_completion_tokens` for Chat and `max_output_tokens` for Responses; `reasoning_effort` maps to the appropriate API field. The Claude preset omits GPT-specific reasoning settings. Chat services may configure `max_tokens_field` as `max_tokens`; the Gemini connection uses that option with `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`. See [Google's OpenAI compatibility API](https://ai.google.dev/gemini-api/docs/openai).

A source slice can specify a lossless `orientation` with `transpose`, `flip_rows`, and `flip_columns`. Slicer derives it from the active view's XY-to-RAS transform and volume's RAS-to-IJK transform. The backend applies the transform before inference and its inverse before merging the result. In-plane rotations needing resampling are rejected explicitly. Tests cover asymmetric arrays and actual coordinates returned through a mock inference endpoint.

### Anthropic Messages

`anthropic-polygons` connects directly to `https://api.anthropic.com/v1/messages` using `x-api-key` authentication and the Messages API version header. It sends PNG image blocks, an explicit model ID and `output_config.format` for structured JSON. `max_output_tokens` maps to `max_tokens`; GPT-specific reasoning settings are rejected.

Segmentation, localization and object classification share the same transport. The adapter expresses unsupported schema limits in field descriptions, then validates the original geometry and label constraints locally. Refused, incomplete or invalid results fail without applying partial output or switching models. See [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

### Adding a provider or training recipe

1. Implement `Segmenter.predict` or `Trainer.train` from `core/ports.py`.
2. Define a small validated configuration for the provider, including dimensional and label expectations.
3. Register it in `server/models/service.py`, or add a recipe resolver to `server/recipes.py`.
4. Record reproducible model state and training lineage. Keep large weights in artifacts or an external model runtime.
5. Test geometry, output validation, cancellation boundaries, and held-out evaluation with that provider.

Training recipes are `pixel-gaussian`, 2D/3D `monai-unet`, and `vista3d` fine-tuning. `pixel-gaussian` is a small diagonal Gaussian classifier over image channels. Scratch starts with zero class statistics; continue adds previously unseen accepted cases; fine-tune weights parent statistics by one-half and adds new cases. Changed labels for previously trained cases require scratch retraining to avoid double-counting. These are recipe-specific semantics, not a universal neural-network fine-tuning strategy.
