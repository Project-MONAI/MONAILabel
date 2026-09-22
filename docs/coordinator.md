# Conversation coordinator

Web, Slicer, QuPath and OHIF send prompts and sample/view context to one backend coordinator. Its language model chooses typed tools; those tools call the same services as UI buttons. Annotation uses the separately selected annotation model. Changing that model does not change the conversation model.

## Setup

The default is local Nemotron Lightning, requiring Docker with NVIDIA GPU support:

```bash
uv run monailabel-server --assistant local --assistant-variant lightning
```

`--assistant-variant 9b` and `4b` select smaller, experimental alternatives. They can misinterpret review filters or choose an incomplete action in multi-step workflows; Lightning remains the default. `--assistant-gpu 1` chooses a GPU when provisioning a runtime. Setup runs in the background and the workspace footer reports readiness. Runtimes and weights are cached under `workspace/.cache/coordinator/`. Only one server may own a workspace.

On Linux ARM64, managed serving uses model-specific shared-memory budgets: Lightning 30%, Nano 4B 15%, and Nano 9B 25%. Other platforms retain an 80% default. All recipes limit concurrency to four requests. Lightning uses a 32,768-token context and total token cap with decode graphs bounded to four sequences; Nano uses a 16,384-token context, eager execution and a 1 GiB KV cache. These are serving budgets, not limits on annotation/training memory. Use one coordinator at a time and leave headroom for the OS, viewers and model workloads. See [Spark validation](spark.md).

Local ARM64 requests default to a 240-second timeout so the allowed reasoning output can finish on GB10. Other platforms and hosted endpoints retain 90 seconds. An explicit `MONAILABEL_ASSISTANT_TIMEOUT` takes precedence. First-start kernel compilation is separate from request timeouts; readiness can take several minutes.

`--assistant-gpu-memory-utilization 0.4` (or `MONAILABEL_ASSISTANT_GPU_MEMORY_UTILIZATION=0.4`) overrides the serving fraction when necessary. A changed managed runtime configuration is rejected until its named cached container is deliberately stopped and removed; weights remain cached. With default GPU/budget settings, a recognized shared Lightning service on port 8001 is reused without changing its memory settings or lifecycle. An explicit memory override or nondefault GPU disables that reuse; stop the external runtime yourself before provisioning another model on the same GPU. Use `--assistant compatible` to select an explicit endpoint. A read-only shared Hugging Face cache is left untouched; managed downloads use the application's writable cache instead.

For a hosted model, set its API key in the server environment, then use:

```bash
uv run monailabel-server --assistant openai --assistant-model YOUR_MODEL
uv run monailabel-server --assistant anthropic --assistant-model YOUR_MODEL
uv run monailabel-server --assistant gemini --assistant-model YOUR_MODEL
# An existing compatible endpoint, local or hosted:
uv run monailabel-server --assistant compatible \
  --assistant-base-url https://your-endpoint/v1 --assistant-model YOUR_MODEL \
  --assistant-key-env YOUR_API_KEY_ENV
```

Built-in credential references are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. Compatible endpoints may omit `--assistant-key-env` when no authentication is required. Runtime settings also accept `MONAILABEL_ASSISTANT_PROVIDER`, `_VARIANT`, `_MODEL`, `_BASE_URL`, `_KEY_ENV` and `_THINKING` with the same prefix.

Local runtime recipes are pinned in `providers/chat/local_models.py`. Managed Lightning and Nano 4B startup are checked on DGX Spark. Windows local serving requires a suitable Linux GPU container environment and separate native platform QA. Hosted adapters have transport contract tests; actual endpoint/account compatibility needs verification.

## Behavior and extension points

- `core/chat.py`: provider-independent messages, calls and execution receipts.
- `providers/chat/`: conversation adapters and managed local runtimes.
- `server/assistant_tools/`: typed setup, annotation, review and learning tools.
- `server/assistants.py`: user/project/sample conversations and bounded history.

Services enforce authorization, geometry, scope, revisions and learning rules. The coordinator has no shell or arbitrary-code tool. Durable request receipts prevent duplicate execution. A returned job or viewer action is pending until its worker or client reports completion.

The coordinator discovers packaged [skills](../packages/server/src/monailabel/server/resources/assistant/skills) for workspace, dataset import, batch annotation, review, training, evaluation, radiology, pathology, video and model onboarding. All skill names and folders use the `monailabel-` prefix. The root `skills/` link opens the same source files. Each follows the [Agent Skills specification](https://agentskills.io/specification). The model sees names and descriptions first, calls `load_skill`, then receives that workflow's instructions and tool schemas. Skill selection resets for each prompt; viewer skills require a selected sample. There is no keyword router.

Use an existing `SKILL.md` as the template for a new workflow: standard `name` and `description` frontmatter, `compatibility` for the required MONAI Label tools, `allowed-tools` listing registered tool names, and `metadata.monailabel-context` set to `workspace`, `project` or `viewer`. Optional `metadata.monailabel-collections` attaches named workspace collections through the read-only inspection tool; dataset import uses it to supply the catalog without asking users for IDs. Keep the body short, with defaults and representative examples. `allowed-tools` narrows model context; it is an experimental Agent Skills field and never grants application permissions. The server loads only its bundled resources, so uploaded files and model repositories cannot install instructions. Restart after editing skills; lint and test the change before distributing the server package.

Conversation history retains requests and recorded outcomes without replaying completed arguments as defaults. Training and comparison can use exact model and evaluation-set names; services resolve IDs and saved structures. Repairs are bounded and cannot substitute a different valid action. Server logs record coordinator finish reasons and token counts, without prompt content or credentials. A response that hits its output limit executes no tool from that response. See the [workflow checks](testing.md).

Skill loading defaults to `purpose=action`. The coordinator must then call an action tool or ask for essential input through `clarify_request`; prose claiming completion is retried within the repair budget and rejected if no action follows. `purpose=answer` supports explanations without performing work.

Batch requests such as “Run VISTA3D segmentation for the first 5 images and submit them for review” run full-image inference in dataset order, skipping evaluation-only cases, saved annotations and pending proposals. Results appear in Reviews as pending; a reviewer must decide whether they are Good. Activity shows per-image progress and any failures, with a live log (latest 1,000 lines) and full-log download. Cancelling keeps already submitted results. Case selection alone only ranks cases; it does not run segmentation.
