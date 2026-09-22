# Verification

Use disposable workspaces for mutation and viewer tests. Never run a second server against an active workspace. The standard lint, type and test commands are in [AGENTS.md](../AGENTS.md#development).

## Workspace browser

Workspace chat and file import can also be checked on an HTTP network hostname, where `crypto.randomUUID` is unavailable:

```bash
uv run --group e2e playwright install chromium
uv run --group e2e pytest --browser-e2e tests/e2e/test_workspace_http.py
```

These checks use Chromium and disposable workspaces with scripted chat responses; they do not call a model. Provider catalog HTTP fixtures exercise NVIDIA, OpenAI, Anthropic and Gemini discovery through the UI, including filtering, search, duplicate detection, provider changes during pending requests, environment references and encrypted saved keys. They also verify a manual custom connection, fixed preset names, visible provider labels, switching a preset to its direct provider and restoring automatic selection. Catalog unit tests cover pagination, upstream failures, capability filtering, project permissions, credential rotation, startup fallback and historical connection preservation.

The radiology Quickstart can also be exercised through a real OHIF build:

```bash
uv run --group e2e pytest --browser-e2e tests/e2e/test_ohif_quickstart.py
```

This test uses a synthetic CT, deterministic VISTA3D/Astra annotation fixtures, and the production OHIF adapter to annotate a volume, correct a slice and submit the result through chat. OHIF is prepared in the disposable cache; `MONAILABEL_OHIF_DIST` can point to an existing build of the current extension. No hosted inference is called.

## Browser desktops

```bash
uv sync --group e2e
uv run playwright install chromium
uv run --group e2e pytest --desktop-e2e tests/e2e/test_browser_desktop.py
```

These opt-in tests require Linux and Docker. They launch real Slicer and QuPath in disposable containers through an HTTP network hostname, verify the full-tab layout, selected sample and submitted annotation, and exchange Unicode clipboard text with a separate browser page. They check that the remote resolution and native application follow browser resizing and tablet orientation changes while preserving draft text. They also refresh without losing the draft and close the last tab to end the session without changing the submitted revision. QuPath exercises tablet-sized touch and keyboard controls and excludes volume-only models. Native probes observe application state; image loading and browser input use the production adapters. No paid models are called. Artifacts are written under `test-results/` and test containers are removed afterward.

The standard suite checks loopback/native versus remote/browser selection, explicit browser override, owner and role enforcement, cross-origin WebSocket rejection, streaming, logout revocation, session limits, reconnect renewal and failed-start cleanup. It also checks status requests during slow setup, continued access to existing desktops, cancellation and account changes during launch, and cleanup when log collection fails. Physical iPad/Safari interaction and GPU rendering performance require separate device checks.

The Slicer exit test submits through the native viewer, chooses **Exit Slicer**, and verifies that its browser tab closes only after the submission is saved. A second tab with closing disabled returns to the project workspace. Lifecycle checks distinguish a confirmed native exit from network failures, unavailable Docker, lost access and obsolete disconnect notifications.

The pathology Quickstart test imports the public OpenSlide sample, draws a native QuPath region through the browser, sends the documented segmentation and submission prompts, and checks that only the selected crop reaches the annotation fixture. QuPath applies the returned objects and publishes the submitted mask through its production adapter.

## Golden prompts

The eight [Spleen learning workflow prompts](workflows.md#try-the-spleen-learning-workflow) come from [spleen.json](../examples/prompts/spleen.json). The runner verifies imports, scoped reviews, batch annotation, named model creation, training and comparison against independent fixed references. The README keeps shorter first-use prompts for each modality.

```bash
uv run python examples/render_golden_prompts.py --check
# Deterministic tools and tiny images; no GPU, network or medical weights:
uv run python examples/verify_spleen_workflow.py --output /tmp/spleen-fixture.json
# Test all eight prompts with the managed Nano 4B conversation model:
uv run python examples/verify_spleen_workflow.py --coordinator 4b --output /tmp/spleen-4b.json
# Real VISTA3D and Decathlon; connect your existing conversation endpoint:
uv run python examples/verify_spleen_workflow.py --real \
  --coordinator-url http://127.0.0.1:8001/v1 \
  --coordinator-model YOUR_SERVED_MODEL \
  --cache-dir workspace/.cache/datasets \
  --output /tmp/spleen-real.json
```

The default runner creates and removes a temporary workspace. Supply `--workspace /path/to/empty-directory` to retain results for inspection. `--cache-dir` reuses dataset downloads; `MONAILABEL_MODELS_DIR` selects a model cache. Omit `--real` to test a conversation endpoint against deterministic inference/training fixtures. Fixture scores verify software behavior, not medical quality. Reports include the instruction revision, loaded skills and tool-call traces. Use `--prompts /path/to/definition.json` to exercise paraphrases and different model names while retaining the workflow's assertions. Repeat runs in fresh workspaces to check consistency; one passing run does not establish reliability.

The real run imports 41 cases, reserves 9 labeled references, annotates 5 of the remaining 32 images, fine-tunes a derived model, and scores it and the unchanged base on identical references. It checks image membership, annotation revisions, model lineage, logs and reports. Reviews are explicitly accepted by the test prompts; this does not replace human inspection in normal use.

Additional [radiology](../examples/prompts/radiology.json) and [pathology](../examples/prompts/pathology.json) definitions exercise viewer actions, model choices and continued U-Net learning:

```bash
uv run python examples/verify_golden_stories.py --story both --output /tmp/golden-stories.json
uv run python examples/verify_golden_stories.py --coordinator lightning --story both --output /tmp/lightning-stories.json
uv run python examples/verify_golden_stories.py --coordinator 4b --story both --output /tmp/4b-stories.json
# Tool selection only; does not execute operational tools:
uv run python examples/evaluate_coordinators.py --variant lightning --output /tmp/lightning-prompts.json
uv run python examples/evaluate_coordinators.py --variant 4b --story endoscopy --output /tmp/4b-endoscopy.json
uv run python examples/evaluate_coordinators.py --variant 4b --suite viewer-edits --output /tmp/4b-edits.json
node --test tests/web/*.mjs
```

## Native viewers

Check Slicer, QuPath and OHIF on disposable images with asymmetric geometry. Verify source orientation, selected-slice scope, class colors, editable hints, stale-revision rejection, submission and corrected review. Preserve user drafts. Headless browser checks do not establish native desktop or physical mobile-device compatibility.

Use `uv run python examples/vista3d_smoke.py --help` for the VISTA3D GPU smoke check. SAM contracts and viewer transfers are exercised in `tests/test_sam.py` and the spatial-hint tests. These exercise runtime and transfer behavior; they are not clinical benchmarks. Remaining capabilities and platform gaps are listed in the [roadmap](roadmap.md).

## Video and CVAT browser tests

The opt-in [video E2E suite](../tests/e2e/test_video_cvat.py) starts the bundled CVAT services and a disposable MONAI Label server, then drives both web applications with Chromium. Install Docker with Compose, FFmpeg (including `ffprobe`), and the browser from the repository root:

```bash
uv sync --locked --group e2e
uv run --locked --group e2e playwright install chromium
uv run --locked --group e2e pytest tests/e2e --video-e2e -v -s --junitxml=test-results/video-e2e.xml
```

On Linux, use `playwright install --with-deps chromium` if browser system libraries are missing. The Docker daemon must be running; the first run downloads CVAT images. The managed tracking cases also download local model weights. No GPU, chat endpoint or paid model is needed. Ordinary `uv run pytest` skips these browser tests.

Each run creates a random Compose project, separate ports, generated credentials and a temporary workspace. Teardown removes that run's containers, volumes and workspace after success or failure. Existing workspaces and CVAT deployments are not used. Screenshots, passed checkpoints, browser errors and redacted service logs remain under the printed `test-results/video-*` directory; the command also writes a JUnit report. The [Video and CVAT E2E workflow](../.github/workflows/video-e2e.yml) runs on relevant pull requests or manual dispatch and uploads these artifacts.

The suite imports a synthetic variable-rate clip through the workspace UI, draws two instrument tracks with the native CVAT mouse controls, and saves occluded/outside keyframes. It checks source timing and geometry, draft resume, immutable submission, separate review tasks, requested changes, stale submission rejection, corrected acceptance, track identity and revision lineage, persistence after server restart, and clip deletion that retains external drafts. A delayed refresh verifies that a submitted correction cannot immediately be reviewed using the previous revision. Additional cases draw an unsupported standalone shape and verify rejection without draft loss, reserve a video procedure from related image training, and exercise annotator access through browser controls and authenticated requests. Annotation writes use CVAT's UI; API reads verify persisted results. These synthetic checks verify software behavior, not annotation quality.

To include the real local coordinator in the 16-frame snare polygon case, set `MONAILABEL_E2E_COORDINATOR_URL` to its OpenAI-compatible API base URL and `MONAILABEL_E2E_COORDINATOR_MODEL` to its served model name. If authentication is needed, `MONAILABEL_E2E_COORDINATOR_KEY_ENV` names the environment variable containing its key. The annotation endpoint remains a deterministic fixture; no hosted annotation calls are made. Without those variables, routing is scripted for reproducibility.

The managed-viewer cases start without CVAT credentials or services, install through the **CVAT** action, and annotate through the embedded native editor using only the workspace sign-in. They run the real local SAM 2.1 tracker, check automatic draft application, reject application after an intervening manual edit, preserve unrelated tracks, exercise embedded chat with scripted coordinator routing, submit and accept revisions, and reopen saved tracks after a server restart. A real HyperKvasir snare sample exercises catalog import, opening CVAT from a workspace tracking request, guidance before a tool box is drawn, and a segmentation request outlining the tool from an empty draft and tracking 16 frames after choosing a vision model and label. A deterministic local HTTP endpoint supplies the vision response; CVAT, source-frame extraction and SAM run normally. Provider contract tests cover Responses and Chat Completions payloads, credentials, abstentions and invalid geometry. An additional case checks vision abstention, an explicitly named model overriding the panel selection, new-track undo/redo and preservation of unrelated manual tracks. These tests download the public clip and pinned model weights; they verify execution and data handling, not clinical tracking accuracy.

A synthetic moving polygon verifies single-frame boxes and segmentation through the configured vision HTTP boundary without a temporal tracker, tracking a selected polygon for N frames, and whole-video tracking from frame 0 while another frame is displayed. A 70-frame clip crosses SAM's chunk boundary. The test inspects automatically added native polygons and downloaded source-size masks, verifies Undo and leaving without submission, saves and reloads mixed box/polygon drafts, and round-trips exact keyframes and identities through a separate review task. It also checks that chat and its composer remain visible at two viewport sizes with no manual settings panel and review controls collapsed until needed. Geometry tests cover border pixels, thin objects, disconnected masks and holes, including warnings and lossless original-mask retention.

A mixed image/video project verifies the shared Datasets table, search, type filters, pagination, selection across pages and refreshes, sample details, and bulk deletion of both media types. All mutation checks run in the disposable workspace.

Viewer launch checks verify that CVAT buttons navigate a loading tab automatically and that chat requests navigate directly to the prepared editor. They also verify cleanup after launch failure, a manual link after closing a loading tab, and automatic navigation when browser popups are blocked.

The smaller video tests in `tests/test_videos.py` cover conversion, revision contracts, authorization and grouping. FFmpeg-dependent tests skip if it is unavailable. `tests/web/videos.mjs` checks capability controls and review filtering without starting services.

## Scoped review and local learning

`tests/e2e/test_scoped_learning.py` submits two regions or two video ranges from one source, inspects their previews in Chromium, accepts one and requests changes to the other, then trains a U-Net through the workspace. It verifies that the snapshot contains only accepted coverage and that training without evaluation produces no held-out score. Run it with `--browser-e2e`.

The native QuPath quickstart test continues from its submitted region to acceptance, real U-Net training and application of that checkpoint in QuPath. The native CVAT polygon clear/undo test continues through the same loop using source video frames. Run those with `--desktop-e2e` and `--video-e2e`, respectively. Small training runs verify integration, not annotation quality.

`tests/test_review_units.py` and `tests/test_optional_evaluation.py` cover immutable scope revisions, accepted coverage, legacy video review migration, held-out grouping and training without evaluation. Loss-gradient checks verify that unreviewed pixels are excluded while project class 255 remains usable.

For GPU and ARM64 setup checks, use `uv run python examples/check_gpu.py` and the [DGX Spark guide](spark.md). Passing on another CUDA machine does not validate GB10 hardware.
