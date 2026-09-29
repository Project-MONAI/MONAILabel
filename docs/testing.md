# Verification

Use disposable workspaces for mutation and viewer tests. Never run a second server against an active workspace. Run the standard [development checks](../AGENTS.md#development) and JavaScript tests:

```bash
node --test tests/web/*.mjs
```

Fixture tests verify software behavior, not model accuracy. Tests below use scripted or local models unless an endpoint is explicitly configured.

## Workspace browser

For real nnU-Net and TotalSegmentator CT/MRI training, continuation and source-grid prediction checks:

```bash
MONAILABEL_TEST_NNUNET_GPU=1 uv run pytest tests/test_nnunet.py
MONAILABEL_TEST_TOTALSEG_GPU=1 uv run pytest tests/test_totalsegmentator.py
uv run --group e2e pytest --browser-e2e tests/e2e/test_nnunet_browser.py
```

```bash
uv sync --locked --group e2e
uv run --group e2e playwright install chromium
uv run --group e2e pytest --browser-e2e tests/e2e/test_workspace_http.py tests/e2e/test_dataset_template_browser.py tests/e2e/test_coordinator_readiness.py
uv run --group e2e pytest --browser-e2e tests/e2e/test_ohif_quickstart.py tests/e2e/test_scoped_learning.py
```

Use `playwright install --with-deps chromium` if system libraries are missing. OHIF builds on first use; `MONAILABEL_OHIF_DIST` can reuse a current build. Annotation responses are fixtures; no hosted inference is called.

## Browser desktops

Requires Linux and a running Docker daemon:

```bash
uv run --group e2e pytest --desktop-e2e tests/e2e/test_browser_desktop.py
```

These checks launch Slicer/QuPath in disposable containers and cover source geometry, editing, submission, session lifecycle, clipboard, dictation insertion and resizing. Artifacts stay under `test-results/`.

OHIF and desktop voice checks inject speech-service events to verify transcript handling, permissions errors and draft preservation over HTTPS. Set `MONAILABEL_E2E_BROWSER_CHANNEL=msedge` to run these checks with an installed Microsoft Edge. They do not verify a physical microphone or the browser's transcription service.

Set `MONAILABEL_E2E_HOST` to an actual local IPv4 address for LAN checks. `MONAILABEL_E2E_LAN_ONLY=1` additionally tests an interface-only listener after local administrator setup. Physical tablets, speech recognition and display performance require device checks.

## Golden prompts

```bash
uv run python examples/render_golden_prompts.py --check
uv run python examples/verify_spleen_workflow.py --output /tmp/spleen-fixture.json
uv run python examples/verify_golden_stories.py --story both --output /tmp/golden-stories.json
```

These use temporary workspaces and deterministic fixtures. `--workspace /path/to/empty-directory` retains results; `--prompts /path/to/definition.json` selects another prompt set.

To test a local conversation model:

```bash
uv run python examples/verify_spleen_workflow.py --coordinator 4b --output /tmp/spleen-4b.json
uv run python examples/verify_golden_stories.py --coordinator lightning --story both --output /tmp/lightning-stories.json
uv run python examples/evaluate_coordinators.py --variant 4b --suite viewer-edits --output /tmp/4b-edits.json
uv run python examples/evaluate_coordinators.py --variant 9b --suite quickstart --output /tmp/9b-quickstart.json
```

`evaluate_coordinators.py` tests tool selection against fixture workspace metadata without executing operational tools. `--suite quickstart` checks every README prompt; `--variant` accepts `4b`, `9b` or `lightning`. Use `--case` to select a prompt (repeat for several), or `--story` to select a golden story. `--provider openai --model gpt-6-astra` tests the hosted assistant using `OPENAI_API_KEY` and incurs API charges. Full training budgets and hosted annotation quality require separate runs.

For real Decathlon/VISTA3D execution with an existing coordinator:

```bash
uv run python examples/verify_spleen_workflow.py --real \
  --coordinator-url http://127.0.0.1:8001/v1 \
  --coordinator-model YOUR_SERVED_MODEL \
  --cache-dir workspace/.cache/datasets \
  --output /tmp/spleen-real.json
```

This downloads data/weights, annotates cases, fine-tunes a derived model and compares it with the base against held-out references. `MONAILABEL_MODELS_DIR` selects the model cache. Evaluation imports accept supplied references automatically; generated annotations still need review.

## Native viewers

Check orientation, selected-slice scope, colors, editable hints, stale-revision rejection and corrected review with asymmetric disposable images. Preserve user drafts.

Model execution checks:

```bash
uv run python examples/check_gpu.py
uv run python examples/vista3d_smoke.py --help
uv run python examples/sam2_smoke.py --frames 70
```

The model checks download cached weights. SAM checks propagation across its 64-frame chunk boundary. Neither requires a second coordinator. Run on the target hardware; see [DGX Spark](spark.md).

## Video and CVAT browser tests

The manual **Video and CVAT E2E** workflow requires a self-hosted Linux runner labeled `nvidia` with Docker and the system prerequisites installed. It uses a disposable workspace.

Requires Docker Compose, FFmpeg/ffprobe and Chromium:

```bash
uv run --locked --group e2e pytest tests/e2e --video-e2e -v -s --junitxml=test-results/video-e2e.xml
```

Tests use separate services, ports and workspaces; teardown removes their containers and volumes. First runs download CVAT images and, for tracking checks, public samples and SAM weights. Screenshots and redacted logs stay in `test-results/video-*`.

The suite covers native editing, draft preservation, tracking, submission/review, revision conflicts and restart persistence. The [CI workflow](../.github/workflows/video-e2e.yml) uploads its artifacts. Ordinary `uv run pytest` skips opt-in browser suites.

To include an existing coordinator in the snare test:

| Environment variable | Value |
| --- | --- |
| `MONAILABEL_E2E_COORDINATOR_URL` | Compatible API base URL |
| `MONAILABEL_E2E_COORDINATOR_MODEL` | Served model name |
| `MONAILABEL_E2E_COORDINATOR_KEY_ENV` | Optional environment variable holding its key |
| `MONAILABEL_E2E_COORDINATOR_TIMEOUT` | Request timeout; default 240 seconds |
| `MONAILABEL_E2E_COORDINATOR_MAX_TOKENS` | Output budget; default 8,192, use 4,096 for managed Nano 4B |

Annotation remains a fixture; CVAT, frame extraction and SAM execute normally.

## Scoped review and local learning

The scoped browser suite accepts one region/range, requests changes to another, and checks that U-Net training uses only accepted coverage. Native QuPath and CVAT tests also apply the resulting checkpoint in the viewer. Run with `--browser-e2e`, `--desktop-e2e` or `--video-e2e` as appropriate.

For release artifacts, run [pip checks](releasing.md#prepare-a-release) and [Docker checks](docker.md#verify-a-build).
