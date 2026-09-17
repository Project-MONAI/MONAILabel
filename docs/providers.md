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

The bundled NVIDIA example uses `switchyard/openai/gpt-5.6-sol` at `https://inference-api.nvidia.com/v1/chat/completions`, using `NV_INFERENCE_API_KEY`. See `examples/models/nvidia-sol.json`. Optional `max_output_tokens` maps to `max_completion_tokens` for Chat and `max_output_tokens` for Responses; `reasoning_effort` maps to the appropriate API field.

A source slice can specify a lossless `orientation` with `transpose`, `flip_rows`, and `flip_columns`. Slicer derives it from the active view's XY-to-RAS transform and volume's RAS-to-IJK transform. The backend applies the transform before inference and its inverse before merging the result. In-plane rotations needing resampling are rejected explicitly. Tests cover asymmetric arrays and actual coordinates returned through a mock inference endpoint.

### Adding a provider or training recipe

1. Implement `Segmenter.predict` or `Trainer.train` from `core/ports.py`.
2. Define a small validated configuration for the provider, including dimensional and label expectations.
3. Register it in `server/models.py`, or add a recipe resolver to `server/recipes.py`.
4. Record reproducible model state and training lineage. Keep large weights in artifacts or an external model runtime.
5. Test geometry, output validation, cancellation boundaries, and held-out evaluation with that provider.

Training recipes are `pixel-gaussian`, 2D/3D `monai-unet`, and `vista3d` fine-tuning. `pixel-gaussian` is a small diagonal Gaussian classifier over image channels. Scratch starts with zero class statistics; continue adds previously unseen accepted cases; fine-tune weights parent statistics by one-half and adds new cases. Changed labels for previously trained cases require scratch retraining to avoid double-counting. These are recipe-specific semantics, not a universal neural-network fine-tuning strategy.
