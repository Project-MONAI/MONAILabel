# Video annotation with CVAT

Import a video, open CVAT from the workspace, and annotate tools with boxes or polygons using assistant chat, a configured annotation model and SAM 2.1 tracking. MONAI Label manages project labels, clip grouping and immutable submitted revisions; CVAT stores editable drafts.

## Installation and launch

Install Docker with Compose and FFmpeg, including `ffprobe`. `./setup.sh` installs FFmpeg and prepares CVAT images when Docker is available. You can also prepare only the pinned CVAT images without starting services:

```bash
uv run monailabel viewer cvat
```

Start MONAI Label normally. The first **CVAT** launch downloads any missing images, starts a private CVAT 2.76.0 service for this workspace, and prepares the browser viewer. Task creation waits for the permissions service to become ready. The CVAT button opens a loading tab that automatically becomes the editor when setup finishes. A launch requested through workspace chat opens the editor in the current tab; blocked popups also use the current tab. Closing a loading tab leaves a manual editor link in the workspace. The integrated viewer uses your MONAI Label sign-in; no separate CVAT account or token setup is needed. Later launches reuse the runtime and saved tasks. Docker must remain running while annotating.

The backend binds to an automatically allocated localhost port. Runtime configuration and browser files live under the workspace cache; generated service credentials are encrypted with the workspace's `secrets.key`. Authenticated proxy routes enforce the video's project permissions, and the browser never receives the service token. Docker volumes retain CVAT drafts across server restarts. Multiple workspaces use separate runtime names and ports.

## Try the tool-tracking sample

In a project, choose **Datasets → Sample datasets → Video → HyperKvasir · endoscopic tool tracking (CVAT)** and import the clip. It adds a **Snare** label and one original 71.28-second, 720 × 576 endoscopy video with 1,782 frames. It includes no reference tracks. Re-importing the sample reuses the clip and preserves submitted annotations.

The clip is the HyperKvasir snare-resection video `99b387e7-d07b-4268-9226-4df450c2a198.avi`, published by Borgli et al. See the [dataset and its license files](https://osf.io/mh9sj/) and [dataset paper](https://doi.org/10.1038/s41597-020-00622-y). The catalog follows the **CC BY-NC 4.0** license supplied in the OSF dataset. Downloads use a pinned file version, size and SHA-256 checksum.

## Annotate, track and review

Videos, images and volumes share the Datasets table, search, filters, pagination and file selection. The Type column identifies each sample and its row offers the supported viewer: CVAT, QuPath, Slicer or OHIF. Use the sample actions menu for metadata and deletion. Video training eligibility is shown in its details; selecting a video does not include it in image training.

1. Import the sample above, or use **Datasets → Import video** for MP4/MOV, Matroska/WebM or AVI files up to 2 GiB. Enter instrument labels and a patient/procedure ID shared by related clips and exported images. **Evaluation only** reserves the procedure from image training; video metrics remain unimplemented.
2. Choose **CVAT**; its editor opens automatically when ready. Every source frame is imported without skipping or resizing. Frame numbers are zero-based; original presentation timestamps are preserved separately.
3. On a frame where the tool is visible, choose an **Annotation model** (for example, a configured GPT vision model), then ask the assistant to locate or segment the named tool. Chat controls the tool label, instance description, shape, model and frame range; no manual settings panel is required. A missing or ambiguous detection creates no proposal. Alternatively, draw a rectangle or polygon using CVAT's **Track** mode, select it, and ask to track it. SAM 2.1 handles propagation across additional frames; its first run downloads pinned weights. CPU execution is supported; CUDA is used when available. Results appear automatically in the editable CVAT draft without an Apply/Discard step. Inspect and correct them in the viewer, or use CVAT **Undo**. Leaving without submitting does not create a review revision. A new track stops after the requested range when another frame exists, and its addition is undoable in CVAT. Tracking an existing annotation preserves its identity and frames outside the requested range. Requesting polygons from an existing box creates a new track and keeps the box. All flows preserve unrelated tracks. If you edited the draft or submitted another revision while inference ran, applying is rejected and your edits remain.
4. Adjust boxes or polygons, keyframes, **Occluded** and **Outside** in CVAT. **Save draft** saves to CVAT. **Submit for review** saves first and then publishes an immutable pending review revision in MONAI Label. Saving alone does not submit or accept annotations.
5. In **Reviews → Video clips**, choose **Inspect in CVAT** for a separate task initialized from the submitted revision. Correct and submit that task before accepting the corrected revision. Review decisions apply to submitted tracks, never unsaved drafts.

The embedded assistant shares the workspace conversation model and typed tools. With the video open, ask:

```text
Use GPT-5.6 Sol to put a bounding box around the snare on this frame.
```

```text
Use GPT-6 Astra to segment the snare on this frame.
```

```text
Use GPT-5.6 Sol to locate the snare and track it for 16 frames.
```

```text
Use GPT-6 Astra to segment the snare entering from the lower right and track it for 16 frames.
```

```text
Segment the snare and track it throughout the whole video.
```

Or select an existing box or polygon and ask:

```text
Track this tool for 16 frames.
```

**Locate/find/bounding box** requests produce boxes; **segment/outline/polygon** requests produce segmentation polygons. If “annotate” is ambiguous, the assistant asks which shape you want. **This frame** produces one annotation without a temporal tracker. **N frames** includes the starting frame. **Whole video** starts at frame 0 even if another frame is displayed; the target must be visible there. For a manual seed, select it on frame 0 first. Long ranges run as a job in chunks of at most 64 frames, carrying the last mask across a one-frame overlap. Progress and **Cancel job** are shown in the panel. A cancelled or failed job applies nothing. Dense polygon output is limited to 2,048 vertices per frame and one million coordinates per proposal; use a shorter range if that limit is reached.

An explicitly named annotation model overrides the panel selection for that request. The conversation model does not implicitly become the annotation model. Compatible configured 2D models can locate or segment the starting frame: GPT vision providers use their registered endpoint, model ID and credential reference; mask providers and local segmentation models use their existing adapters and supported labels. A model must support the selected label and 2D source frame. SAM 2.1 is currently the only temporal tracker. It propagates the chosen model's initial annotation; GPT does not independently process every subsequent frame. Selecting or opening a model does not invoke it. Proposals record the annotation model and, when used, tracker provenance. Inspect and correct the result before submitting.

Polygons approximate source-grid segmentation masks. Each frame uses its largest outer contour; holes or disconnected regions produce a visible warning. **Segmentation details → Download original masks** in the chat retains all segmented pixels as lossless source-size PNGs, named by source frame, with metadata in a ZIP. These masks are retained with the proposal; submitted CVAT revisions contain the editable polygon geometry. Complex masks therefore remain available without implying the polygon contains every segmented pixel.

If a visible, unlocked rectangle or polygon track is selected, an ordinary tracking request follows it using SAM. With no track selected, a selected annotation model and tool label enable automatic annotation followed by tracking. Otherwise the assistant explains both choices. Asking from workspace chat opens CVAT when the project has one clip; with multiple clips, choose the intended clip's CVAT action. Repeat the request inside CVAT after choosing a starting frame. Tracking from a manual annotation needs no remote vision model; the assistant still requires its conversation model. Video training, automatic rediscovery after tracking loss and tracking metrics remain [planned work](roadmap.md).

Reopening the same mode and revision resumes its saved CVAT task without replacing annotations. Reopening an older task from the workspace also enables polygon labels while retaining its label IDs and annotations; save unsaved edits before reopening. After submission, opening again creates a task for the new revision. Stale submissions are rejected while retaining the older draft. Avoid editing the same CVAT task in multiple tabs: submission uses its last saved state.

Track identities, fractional edge coordinates, rectangle and polygon keyframes, occlusion/outside states and native interpolation survive the round trip. Unsupported tags, standalone shapes, rotated rectangles, groups, attributes and nonzero z-order are rejected without silently dropping them. Deleted/filtered frames, changed dimensions and mismatched frame counts block submission. Convert rotated clips to upright pixels before import.

## API and data retention

The authenticated API exposes video import/listing, original source download, frame metadata, track revisions, editor jobs, annotation/tracking proposals, original proposal masks, submission and review. See the running server's `/docs` for schemas. Video records never enter image training snapshots. Metadata stores presentation-order `timestamps` relative to the first frame and the original `start_time`; original timestamps are `start_time + timestamps[i]`. Boxes use `[left, top, right, bottom]`; polygon points are flat `[x1, y1, x2, y2, ...]` source image edge coordinates. Grouping must be supplied correctly; differently encoded duplicates cannot be inferred automatically.

Use **Delete file** in sample actions, or select clips and images together and choose **Delete selected files**. Deletion removes workspace sample records and submitted annotations subject to active-job and learning-history checks. Project deletion removes its workspace records. CVAT tasks and Docker volumes are retained, including saved drafts, and require separate retention management. The [browser tests](testing.md#video-and-cvat-browser-tests) use disposable services and workspaces.

## Optional external CVAT service

Use an existing CVAT deployment when you need to administer its accounts separately. External mode opens CVAT's own UI; the embedded MONAI Label panel belongs to managed mode.

Install FFmpeg and Docker Compose. The optional external service uses CVAT 2.76.0 with its database, workers and browser UI. From the repository root:

```bash
sudo apt-get install ffmpeg
export MONAILABEL_CVAT_COMPOSE=packages/viewers/src/monailabel/viewers/resources/cvat/compose.yaml
docker compose -f "$MONAILABEL_CVAT_COMPOSE" up -d
docker compose -f "$MONAILABEL_CVAT_COMPOSE" exec server python manage.py createsuperuser
```

Wait for the server to initialize before creating the account. Choose your own username and password. Open **http://localhost:8093** and sign in. The backend port is **8092**; **8093** routes both the UI and API. `MONAILABEL_CVAT_API_PORT` and `MONAILABEL_CVAT_UI_PORT` override these ports when starting Compose. These services bind to localhost and keep their data in Docker volumes.

Configure the MONAI Label server with a CVAT origin and authentication token. This command requests a token using hidden password input without printing it or putting the password in shell history:

```bash
export MONAILABEL_CVAT_URL=http://localhost:8093
read -r -p 'CVAT username: ' MONAILABEL_CVAT_USERNAME
export MONAILABEL_CVAT_USERNAME
MONAILABEL_CVAT_TOKEN="$(uv run python - <<'PY'
import getpass
import os
import httpx

username = os.environ['MONAILABEL_CVAT_USERNAME']
password = getpass.getpass('CVAT password: ')
response = httpx.post(
    os.environ['MONAILABEL_CVAT_URL'] + '/api/auth/login',
    json={'username': username, 'password': password},
    timeout=30,
)
response.raise_for_status()
print(response.json()['key'])
PY
)"
export MONAILABEL_CVAT_TOKEN
uv run monailabel-server
```

Keep the token in the server environment or your deployment's secret manager. Do not put it in project instructions or chat. `MONAILABEL_CVAT_PUBLIC_URL` optionally specifies a different origin reachable by the browser; the backend uses `MONAILABEL_CVAT_URL`. Neither URL accepts embedded credentials. Restart MONAI Label after configuring them. If both are absent, MONAI Label manages CVAT automatically. Set both when using an external service.

CVAT has its own login and authorization. The account in the browser must have access to the task created by the configured token's account. A local single-user setup can use the same account for both. For a team deployment, administer CVAT task access separately; MONAI Label roles do not provision CVAT accounts or provide SSO. This bundled localhost deployment is intended for a trusted local environment.
