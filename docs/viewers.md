# Viewers

Open samples from **Datasets** to annotate or **Reviews** to review.

| Data | Viewer |
| --- | --- |
| Scalar volumes | Slicer |
| Bounded pathology images | QuPath |
| DICOM or scalar NIfTI | OHIF |
| Video | CVAT |

## Local and remote launch

Slicer/QuPath open native windows for localhost access and private browser desktops for network access. OHIF/CVAT always open in the browser. Use `/datasets?desktop=browser` for a headless host or SSH forwarding.

Browser desktops resize with the tab. The side toolbar provides a keyboard, touch controls and **Clipboard → Paste into viewer / Copy to computer**.

Submit or save drafts before closing. Closing the last tab ends the desktop after ten seconds; **End session** closes it immediately. Refresh and network interruptions preserve the session. Server restarts retain running containers, but crashes/reboots lose unsaved native state. Saved desktop files remain under `workspace/viewers/desktop/`.

### Browser desktop server setup

Requires Linux x86_64/ARM64 and a local Docker daemon. See [Spark](spark.md) for ARM64 viewers and [Docker deployment](docker.md) for running the backend in a container.

Configure HTTPS and allowed hosts for remote access. Reverse proxies must forward WebSocket upgrades for `/desktop/`. The viewer bridge must reach and trust the backend URL; `MONAILABEL_DESKTOP_BACKEND_URL` overrides it. For private development certificates, terminating TLS at the proxy avoids installing the CA in each viewer container.

| Setting | Default |
| --- | --- |
| `MONAILABEL_DESKTOP_LIMIT` | 8 desktops |
| `MONAILABEL_DESKTOP_MEMORY` | 8g per desktop |
| `MONAILABEL_DESKTOP_CPUS` | 4 per desktop |

Desktops use host networking and software rendering. They are intended for trusted workspace users who can run native viewer scripts. Project access and session ownership are checked during display connections.

## Annotate and review

Choose a compatible model and name the target and scope. Inspect and correct results before submitting. Submission creates a pending revision; reviewers accept, request changes or accept a corrected revision. **Reviews → Review decision** also supports decisions on selected rows.

## Slicer

Use **Manual editing** for Segment Editor and **Ctrl+Enter** for chat. Select the intended source-aligned Red/Yellow/Green view for slice requests.

```text
Clear spleen on this slice.

Undo my last change.

Change spleen color to blue.
```

Save a Slicer scene to retain native boxes, points and ROIs. After adapter updates, save/submit, end the session and reopen.

For localization, name a capable model: `Add ROI for spleen between slice 70 to 80 using Astra`. ROI slice numbers are one-based and inclusive, not millimeter positions. Creating an ROI does not automatically constrain later segmentation.

## QuPath

Use the **MONAI Label** tab for chat. A selected region is processed as one crop; whole-image requests can use tiles, defaulting to 256 pixels. Outside edits are preserved.

```text
Annotate nuclei in the selected region.

Clear nuclei in the selected region.

Undo that.
```

**Automatic** uses compatible defaults or the configured Astra preset. Name another model explicitly if needed. Classify native objects as project structures before mask submission; selection guides are not labels. **Save QuPath draft** preserves the native project.

`Classify the annotated nuclei` proposes editable categories for up to 128 objects. Durable cell-type review/training and streaming whole slides are not implemented.

## OHIF

Saved masks load automatically. A failed load offers Retry and blocks submission. Use native tools or chat for slice edits, Undo/Redo and submission. **Return to workspace** reuses the launch page.

NIfTI viewing uses a cached DICOM copy; annotation stays on the original grid. See [DICOM import and viewing](#dicom-import-and-viewing).

## CVAT

Requires Docker Compose and FFmpeg. See [video annotation](video.md) for import, box/polygon tracking, draft saving and review. Managed CVAT uses the workspace sign-in. Saved drafts and submitted review revisions are separate.

## Colors

Project colors are shared across viewers. New anatomical labels use Slicer's palette; manual overrides are preserved. See [palette source and license](../packages/core/src/monailabel/core/resources/README.md).

## Provisioning and platforms

```bash
uv run monailabel viewer slicer
uv run monailabel viewer qupath
uv run monailabel viewer ohif
uv run monailabel viewer cvat
export MONAILABEL_SLICER_EXECUTABLE=/path/to/Slicer
```

Tools cache under `workspace/.cache/tools/`; override with `MONAILABEL_TOOLS_DIR`. Standalone CLI provisioning uses the OS user cache and launches on that machine. Use `--url https://your-server` for a remote backend.

OHIF's first build needs Node.js 22, Corepack and Git. `MONAILABEL_OHIF_DIST` selects an existing build. Linux is supported; native Windows/macOS validation remains pending.

## Interactive segmentation

nnInteractive segments prompted objects in CT/MRI volumes. MedSAM2 propagates a seed slice through a medical volume; SAM 2.1 handles 2D images and single slices. These models need user points or a box to identify the object.

### Setup

First use downloads the model: about 411 MB for nnInteractive or 156 MB for each SAM model. nnInteractive also installs its pinned inference environment. `MONAILABEL_MODELS_DIR` overrides the cache.

### Radiology

Open an image in Slicer or OHIF and send:

```text
Start nnInteractive for spleen.
```

Choose a **Model** and select or type a **Target label**. The Model menu separates radiology, interactive and vision-language models, followed by Trained models. Only compatible, nonempty groups appear; headings cannot be selected. The selected model determines which input tools are shown. The toolbar uses icons with tooltips. Choose **+ Point** to mark tissue to include, **− Point** to exclude tissue, or **Box** to draw around it. Slicer boxes use two opposite-corner clicks; OHIF boxes use a drag. Points and box handles remain editable. Selecting a model does not start drawing until you choose a tool. Point mode stays active while scrolling between slices.

Use **▶ Update slice** or choose **Update volume** from its arrow menu. SAM 2.1 supports only **Update slice**. In Slicer, the colored **Slice view** icon opens a menu for Red, Yellow or Green view and sits before the input tools; separators keep the view selector, tools and Update action distinct. nnInteractive combines points across slices and one slice box. MedSAM2/SAM use hints on the selected seed slice. A selected box resolves multiple matching boxes. Inspect the result, adjust hints, and press **Update** again. Each update replays the current hints; it does not use an unsaved mask as an additional model prompt.

Switch **Target label** to annotate another structure. Each label retains its own hints across slices; only the active label’s hints are shown, while all masks stay visible. Update changes only the active label and preserves other labels.

Click the active input tool again, press **Esc**, or say “Stop interaction mode” to stop drawing and retain hints and masks. Models without interactive inputs keep the view and Update controls. Switching models preserves hints; only interactive models use them during inference. These chat controls also work:

```text
Switch to negative points.

Draw a spleen box.

Clear negative points on this slice.

Clear all spatial hints.

Stop interaction mode.
```

Clearing hints does not erase segmentation. Changing hints, the source slice, the image or the mask during inference prevents the stale result from being applied. Press **Update** again after editing. Save a Slicer scene to retain hints; OHIF hints last for the session.

Use `Start MedSAM2 for spleen.` for volume propagation, or `Start SAM 2.1 for spleen.` for a selected slice. Prompts can also run inference directly: `Segment spleen in the whole volume with nnInteractive.`

### Pathology and other 2D images

In QuPath, choose SAM 2.1 and a target label, then add positive/negative points or drag a box around one object. Use **Update image**, or select a region and use **Update region**. You can also start drawing through chat:

```text
Start SAM 2.1 for nucleus.
```

Hints are kept separately for each target and saved with the QuPath draft. “Clear all input points and boxes” removes hints without deleting annotation objects. SAM 2.1 annotates one object at a time; it does not perform automatic whole-slide nuclei detection.

### Video

In CVAT, select SAM 2.1, a target label and **New object** or an existing object. Add positive/negative points or drag a box, then use **Update frame** or **Track range**. Inputs stay separate for each object, label and frame. Say “Clear negative points on this frame” or “Clear all inputs in the whole video” to remove them. Clearing inputs preserves tracks. CVAT retains editable polygons and the backend retains the original lossless masks.

### Provenance and execution limits

| Model | Checkpoint |
| --- | --- |
| SAM 2.1 tiny | [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5) |
| nnInteractive v1.0 | [MIC-DKFZ/nnInteractive](https://huggingface.co/MIC-DKFZ/nnInteractive/tree/3f308d751c00644e4fde6f09c600264b393b21b5/nnInteractive_v1.0); CC BY-NC-SA 4.0 (noncommercial, attribution, share-alike) |
| MedSAM2 | [wanglab/MedSAM2](https://huggingface.co/wanglab/MedSAM2/tree/e4a6f35edd7e091619cbc0750f462f1574e23955), `MedSAM2_latest.pt` |

Runtime/configuration sources and licenses are recorded in [third-party notices](../THIRD_PARTY_NOTICES.md). Fine-tuning these interactive models and automatic evaluation without spatial prompts are unsupported. Held-out masks must not supply evaluation prompts.

## DICOM import and viewing

### Connect, filter and import

1. Open **Datasets → Import from DICOM server**.
2. Enter a backend-reachable DICOMweb endpoint, such as `http://localhost:8042/dicom-web`, and configure authentication.
3. Search by modality, date, import status, patient/accession, descriptions or exact UIDs.
4. Select series or **Import all matches**. Searches are capped at 1,000 series; refine filters if needed.

The endpoint must support QIDO `/series` search with pagination and WADO instance retrieval. PACS/DIMSE addresses and Orthanc REST roots are not DICOMweb endpoints. Credentials are encrypted in the workspace.

Activity reports imported, skipped and failed series. Cancellation retains completed imports; retries skip them. Incomplete series do not become assets.

### Where images are served

OHIF reads imported workspace copies through authenticated DICOMweb routes; viewing no longer requires the source server. Original headers and UIDs are retained. Patient ID, or study ID when absent, supplies the source group. Check grouping before training/evaluation.

### Open NIfTI in OHIF

Choose **OHIF** on a 3D sample. First use creates a cached Secondary Capture viewing series with synthetic identifiers and modality OT; no DICOM server is required. Annotations map back to the original NIfTI grid. Values outside signed 16-bit range use quantization and rescale metadata in the viewing copy.

### Limits

- DICOM: regular scalar single-frame CT/MR; up to 3,000 instances and 256 MiB per series. Enhanced multiframe, irregular spacing, gantry tilt and unsupported decoders are rejected.
- NIfTI: scalar 3D with orthogonal voxel axes, including oblique rotations. Sheared grids need external resampling or Slicer.
- Local DICOMweb serves the bundled viewer; DICOM SEG export is not implemented.
- Original files can retain patient-identifying headers; protect workspace access.

An optional [Orthanc example](../deploy/dicom/compose.yaml) provides an import source. The SeriesInstanceUID import uses `MONAILABEL_ORTHANC_URL`.

## Voice and remote access

**Microphone** transcribes into the prompt box; inspect and send it manually. **Read replies** enables spoken responses. Recognition uses the browser's speech service and may send audio to that provider. MONAI Label stores prompt text, not microphone audio. Unsupported browsers retain typing.

In remote Slicer and QuPath, click the viewer's prompt box, then open **Dictate a prompt** (microphone icon) in the left browser-desktop toolbar. Choose **Use microphone**, speak, and review the transcript. **Insert into viewer** pastes it at the cursor; use the viewer's Send button when ready. OHIF has its microphone beside the assistant's Send button. All use the microphone on your browser device.

Safari requires Siri and microphone/speech permission. See [browser support](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition) and [WebKit setup](https://webkit.org/blog/11648/new-webkit-features-in-safari-14-1/).

### Phones, tablets and other computers

Start with `--host 0.0.0.0 --port 8000`, or bind a specific interface. Create the administrator from localhost first. Connect using the server's hostname/IP; phone `localhost` refers to the phone.

`MONAILABEL_ALLOWED_HOSTS` replaces the default hostname/IP allowlist for custom aliases or proxies; it does not change the listen address. Allow the chosen port through the firewall.

### HTTPS

Stop the running server, then start with:

```bash
uv run monailabel --host 0.0.0.0 --https
```

Open `https://YOUR_SERVER_IP:8000`. Certificates are generated under `workspace/.tls/` and reused on restart. Localhost, the server hostname and detected LAN address are covered; custom aliases can be supplied through `MONAILABEL_ALLOWED_HOSTS` before startup. Managed Slicer and QuPath sessions trust this certificate automatically.

For voice input, copy **only `workspace/.tls/ca.crt`** to the browser device and import it into the browser or operating system's trusted certificate authorities, then restart the browser. Keep `ca.key` and `server.key` private. Clicking through an untrusted certificate warning is not a substitute for this trust step. No trust-store changes are made automatically.

For an existing certificate, use:

```bash
uv run monailabel --host 0.0.0.0 --ssl-certfile server.crt --ssl-keyfile server.key
```

On small screens, use Menu/Assistant controls; OHIF side panels start collapsed.

### Verification limits

Test microphones and touch behavior on the target device. Headless browser checks do not establish physical iOS/Android compatibility. See [verification commands](testing.md).
