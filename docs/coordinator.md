# Conversation coordinator

The coordinator interprets prompts and calls typed workspace tools. Configure annotation models separately under **Models**.

Assistant replies display Markdown tables, lists and code. Drag the assistant's left edge to resize it; the browser remembers the width. A focused divider also accepts arrow keys. Ask for a model's default training parameters to inspect its recipe settings.

## Setup

Local Nemotron Lightning is the default:

```bash
uv run monailabel --assistant local --assistant-variant lightning
```

Use `--assistant-variant 4b` or `9b` for experimental smaller models. The workspace footer reports readiness and enables chat after a small tool-calling check succeeds; first startup can take several minutes. The check does not execute workspace actions. Weights are cached under `workspace/.cache/coordinator/`.

| Setting | Use |
| --- | --- |
| `--assistant-gpu 1` | Select the device when provisioning |
| `--assistant-gpu-memory-utilization 0.4` | Override the serving fraction |
| `MONAILABEL_ASSISTANT_TIMEOUT` | Override request timeout; defaults to 240 seconds for local ARM64, 180 for Responses assistants and 90 otherwise |
| `--no-assistant-thinking` | Disable reasoning where supported |

ARM64 serving fractions are 30% for Lightning, 15% for Nano 4B and 25% for Nano 9B; other platforms default to 80%. Leave memory for viewers, annotation and training. Run one coordinator at a time.

Changing a managed runtime's configuration requires stopping and removing its named cached container; weights remain cached. Default settings can reuse a recognized shared Lightning service on port 8001. Selecting another GPU or setting a memory override disables that reuse. See [Spark setup](spark.md).

An HTTP 5xx error means the model service failed while processing the request. Check that service's health and runtime logs, then retry once it is healthy.

For hosted or existing endpoints, set the API key in the server environment:

```bash
uv run monailabel --assistant openai --assistant-model gpt-6-astra
uv run monailabel --assistant anthropic --assistant-model YOUR_MODEL
uv run monailabel --assistant gemini --assistant-model YOUR_MODEL

uv run monailabel --assistant compatible \
  --assistant-base-url https://your-endpoint/v1 --assistant-model YOUR_MODEL \
  --assistant-key-env YOUR_API_KEY_ENV
```

Built-in key variables are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. Omit `--assistant-key-env` for unauthenticated compatible endpoints. Environment configuration uses `MONAILABEL_ASSISTANT_` followed by `PROVIDER`, `VARIANT`, `MODEL`, `BASE_URL`, `KEY_ENV`, `THINKING` or `GPU_MEMORY_UTILIZATION`.

With only `OPENAI_API_KEY`, GPT Astra runs the assistant and the predefined Astra annotation model connects directly to OpenAI. No NVIDIA API key or local Nemotron service is needed. OpenAI assistants use the [Responses API](https://developers.openai.com/api/docs/guides/function-calling). Hosted assistants must pass a tool-calling check before chat becomes ready; account credit, quota or authentication errors keep chat disabled and show the cause.

For GPT Astra through the NVIDIA gateway, use `NV_INFERENCE_API_KEY` and its Responses endpoint:

```bash
uv run monailabel --assistant openai \
  --assistant-base-url https://inference-api.nvidia.com/v1 \
  --assistant-model azure/openai/gpt-6-astra \
  --assistant-key-env NV_INFERENCE_API_KEY
```

The key must have access to that gateway model. This uses NVIDIA gateway billing, separate from direct OpenAI API billing.

## Behavior and extension points

| Code | Responsibility |
| --- | --- |
| `core/chat.py` | Messages, tool calls and execution receipts |
| `providers/chat/` | Conversation adapters and local model recipes |
| `server/assistant_tools/` | Typed actions |
| `server/assistants.py` | Conversations, history and execution |

Services enforce permissions, geometry and revisions. Request receipts prevent duplicate execution. Jobs remain pending until completion; output-limit responses execute no tool. Logs omit prompt content and credentials.

Workflow [skills](../packages/server/src/monailabel/server/resources/assistant/skills) are bundled with the server. To add one, copy an existing `SKILL.md`, use a `monailabel-` name, declare registered `allowed-tools`, and set `metadata.monailabel-context` to `workspace`, `project`, `project-workspace` or `viewer`. Use `project-workspace` for actions that must stay in the main web workspace, such as annotation batches. Optional `metadata.monailabel-collections` exposes read-only collections. Restart and run [checks](testing.md) after editing.

`allowed-tools` limits model context; it does not grant permissions. Uploaded data cannot install skills. Skill loading uses `purpose=action` for tool execution and `purpose=answer` for explanations. Actions require a tool call or `clarify_request` for missing input.
