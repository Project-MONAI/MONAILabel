# Video annotation with CVAT

## Installation and launch

Install Docker Compose and FFmpeg/ffprobe. Prepare CVAT images with `./setup.sh` or:

```bash
uv run monailabel viewer cvat
```

Open a video's **CVAT** action. First launch prepares a workspace-specific service and browser editor using your MONAI Label sign-in. Keep Docker running while annotating. Saved drafts persist in Docker volumes across server restarts.

## Try the tool-tracking sample

Choose **Datasets → Sample datasets → Video → HyperKvasir · endoscopic tool tracking (CVAT)**. The sample includes a Snare label and a 71.28-second, 720 × 576 clip with 1,782 frames, without reference tracks. Re-importing preserves submitted annotations.

The OSF release supplies [CC BY-NC 4.0 terms](https://osf.io/mh9sj/). Cite Borgli et al., [HyperKvasir](https://doi.org/10.1038/s41597-020-00622-y). For labeled still images, see [Kvasir-Instrument](datasets.md#labeled-endoscopy-and-video).

## Annotate, track and review

1. Import the sample, or use **Datasets → Import video** for MP4/MOV, Matroska/WebM or AVI up to 2 GiB. Labels are optional at project creation and import; name the object when annotating. Use a patient/procedure ID shared by related clips and images.
2. Open CVAT and select a frame where the object is visible. Ask to locate it as a box or segment it as a polygon. A new object name adds a project label; the viewer saves its draft and refreshes before annotation. Alternatively, draw and select a rectangle/polygon in **Track** mode.
3. Request tracking if needed. SAM 2.1 propagates the seed; first use downloads weights.
4. Inspect and correct the result. **Save draft** saves to CVAT. **Submit for review** saves and publishes a pending revision in MONAI Label.
5. In **Reviews**, inspect each frame range and choose Good or Needs changes. Correct and resubmit changed ranges before reviewing the new revision.

```text
Locate the snare on this frame using GPT Astra.

Segment the snare and track it for 16 frames.

Track this selected tool for 16 frames.

Segment the snare throughout the whole video.

Clear the snare annotations for 16 frames.

Undo that.

Submit this annotation for review.
```

**This frame** uses no temporal tracker. **N frames** includes the starting frame. **Whole video** starts at frame 0; the target or selected manual seed must be visible there. Long jobs show progress and cancellation controls.

A selected visible, unlocked track is used as the seed. Otherwise, the configured annotation model finds the named target. Explicit model names override panel selection; compatible project defaults take precedence over the Astra preset. SAM remains the temporal tracker.

Results appear in the editable draft automatically. Unrelated tracks and frames outside the requested range are preserved. Intervening draft/revision changes reject application. Failed or cancelled tracking applies nothing. Use CVAT Undo for applied changes.

Polygons use the largest outer contour of each mask; holes/disconnected regions trigger a warning. **Segmentation details → Download original masks** exports the lossless source-size PNG masks and metadata. Polygon limits are 2,048 vertices per frame and one million coordinates per proposal; shorten the range if needed.

Use rectangle/polygon tracks. Standalone shapes, rotated boxes, tags, groups, attributes and nonzero z-order are unsupported. Changed/filtered frames or mismatched dimensions block submission. Convert rotated videos to upright pixels before import. Avoid editing one task from multiple tabs; submission uses its last saved state.

## Train a tool segmentation model

Accept polygon ranges, create a U-Net for the tool, then train from approved samples. One video can train without evaluation; percentage evaluation needs at least two independent procedure groups. Frames from one procedure stay together. Boxes alone cannot train segmentation. See [learning rules](workflows.md#review-regions-and-frame-ranges).

Name the trained model in CVAT to segment later frames. SAM still handles requested temporal propagation; this workflow does not train a tracker.

## API and data retention

See `/docs` on the running server for video, frame, track, proposal and review schemas. Frames are zero-based. Original timestamps are `start_time + timestamps[i]`. Boxes use `[left, top, right, bottom]`; polygons use flat source pixel-edge coordinates `[x1, y1, ...]`.

**Evaluation only** excludes the procedure from all training. Tracking metrics are not implemented. Supply grouping explicitly; differently encoded duplicates cannot be inferred reliably.

Deleting samples checks active jobs and learning history. CVAT tasks and volumes, including drafts, remain and require separate retention management.

## Optional external CVAT service

External mode uses CVAT's own UI and accounts. The embedded assistant is available in managed mode.

```bash
sudo apt-get install ffmpeg
export MONAILABEL_CVAT_COMPOSE=packages/viewers/src/monailabel/viewers/resources/cvat/compose.yaml
docker compose -f "$MONAILABEL_CVAT_COMPOSE" up -d
docker compose -f "$MONAILABEL_CVAT_COMPOSE" exec server python manage.py createsuperuser
```

Wait for initialization before creating the user. The UI is at `http://localhost:8093`; backend port 8092. Override them with `MONAILABEL_CVAT_UI_PORT` and `MONAILABEL_CVAT_API_PORT` when starting Compose.

Request a token without putting the password in shell history:

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
uv run monailabel
```

Keep the token in the server environment or secret manager. Set `MONAILABEL_CVAT_PUBLIC_URL` when the browser uses a different origin; neither URL accepts embedded credentials. Restart after configuration. With both URLs absent, MONAI Label manages CVAT.

The browser account needs access to tasks created by the token's account. Administer external CVAT permissions separately; MONAI Label roles do not provision CVAT accounts or SSO.
