# Viewers

Open a sample from **Datasets** to annotate, or from **Reviews** to review it. Viewers share the project's models, labels, revisions and backend assistant.

| Data                           | Viewer              |
| ------------------------------ | ------------------- |
| Scalar NIfTI / CT volumes      | 3D Slicer           |
| Bounded RGB pathology images   | QuPath              |
| Imported DICOM or scalar NIfTI | OHIF in the browser |
| Video / endoscopy clips        | CVAT in the browser |

Server-launched desktop tools are discovered first, then downloaded if supported, and cached in `workspace/.cache/tools/`. `MONAILABEL_TOOLS_DIR` overrides this path; standalone CLI provisioning uses its OS user cache. OHIF is prepared separately on first use. NIfTI viewing in OHIF does not require a DICOM server.

## Annotate and review

Choose a compatible model and state the target and scope, such as `Annotate spleen on this slice` or `Annotate nuclei in the selected region`. Inspect and correct the result with native tools before submitting. A single slice or small region is not a complete volume/image annotation.

Submission creates a pending annotation. Reviewers can accept, request changes, or accept corrections in the viewer. Corrected acceptance saves a new immutable revision and its decision together. The web Reviews table also supports selected saved annotations: select one or more rows, then use the single **Review decision** dropdown above the table for **Good / Needs changes / Pending**.

Successful submission/review offers a return or exit action. Keep working to retain unsaved viewer content. Failures leave the viewer open. The workspace refreshes committed changes automatically while protecting open forms.

## Slicer

Use **Manual editing** for Segment Editor and **Ctrl+Enter** to send chat. The conversation panel can be resized or popped out. Select the intended Red/Yellow/ Green view for slice requests; geometry must align with a source voxel plane.

```text
Clear spleen on this slice
Undo my last change
Change spleen color to blue
Restore the default anatomical color for spleen
```

Local mask edits preserve other structures and requested-outside scope. Save a Slicer scene to retain boxes, points, ROIs and other native edits. Reopen the viewer after bridge updates. Local launches can read authorized workspace files directly; remote launches use authenticated downloads.

For model-assisted boxes, explicitly select a capable localization model. `Add ROI for spleen between slice 70 to 80 using Sol` evaluates all eleven slices. ROI slice numbers are one-based and inclusive; they are not Slicer's millimeter position. ROI creation does not automatically constrain later segmentation. See [SAM](#local-sam-annotation) for zero-based point/box editing prompts.

## QuPath

The **MONAI Label** tab provides chat and optional pop-out controls. A selected area runs as one crop without tiling; whole-image requests can specify a tile size (default 256 pixels). Results return to source coordinates and preserve outside edits.

Classify native objects as project structures before mask submission. Selection guides are not segmentation labels. **Save QuPath draft** preserves the native project; save before closing or reopening after an adapter update.

`Classify the annotated nuclei` proposes editable cell categories for up to 128 objects. Classification is separate from the segmentation mask. Backend cell-type review/training and streaming gigapixel slides are not implemented.

## OHIF

Saved masks load automatically with the selected source image, including while the assistant is collapsed. Opening/closing the panel preserves local mask edits and chat. A failed mask load offers Retry and blocks submission until ready.

Use native editing or chat, then submit. **Return to workspace** reuses the launch page when possible. NIfTI uses a cached local DICOM viewing copy while annotations stay on the original grid. See [DICOM import and viewing](#dicom-import-and-viewing).

## CVAT

The [video workflow](video.md) imports clips, opens CVAT directly into a rectangle/polygon annotation job and submits saved tracks for workspace review. It requires FFmpeg and Docker Compose. The workspace prepares CVAT on first launch and opens an integrated viewer using the workspace sign-in, with an assistant-focused panel. Configured annotation models locate or segment tools on a source frame; local SAM 2.1 tracks them over a requested range or whole clip. Describe the tool, shape and frame range in chat. `uv run monailabel viewer cvat` can download the images in advance. Saved drafts and submitted revisions are distinct; review tasks start from the submitted revision.

## Colors

Labels share project display colors across viewers. New anatomical labels use Slicer's Generic Anatomy palette, with user overrides preserved. DICOM stores a recommended display color per segment; it does not define one universal organ palette. See [palette provenance](../packages/core/src/monailabel/core/resources/README.md).

## Provisioning and platforms

```bash
uv run monailabel viewer slicer
uv run monailabel viewer qupath
uv run monailabel viewer ohif
# Use an existing Slicer installation:
export MONAILABEL_SLICER_EXECUTABLE=/path/to/Slicer
```

Linux desktop flows are supported. QuPath has a Windows portable installer recipe; Slicer on Windows and viewers on macOS can use explicit existing executables. Native Windows/macOS QA and automatic Slicer Windows installation remain pending.

OHIF's first source build needs Node.js 22, Corepack and Git. Set `MONAILABEL_OHIF_DIST` to use an existing distribution. Pinned installer recipes live in `viewers/resources/installers/`.

Browser desktop launching assumes the server is on that desktop. For a remote backend, log in and launch through the CLI on the viewer machine using `--url https://your-server`. Network access requires appropriate allowed hosts and HTTPS; see [remote access](#voice-and-remote-access).

## Local SAM annotation

**Implemented:** SAM 2.1 for 2D images / individual volume slices, and MedSAM2 for medical volumes. Both run locally, appear under **Models → Annotation**, and use immutable, checksum-verified weights cached under `workspace/.cache/models/`. New projects default to VISTA3D; select SAM explicitly to use box/point prompts.

### Setup

```bash
uv run monailabel-server
```

The server includes the MONAI and SAM runtimes. First inference downloads the selected checkpoint (about 156 MB each). Set `MONAILABEL_MODELS_DIR` to change the shared cache and `MONAILABEL_SAM_DEVICE=cpu` or `cuda:0` to choose execution. CUDA is selected when available. Native Windows runtime/viewer QA is still pending.

SAM is spatially prompted: **a target name identifies the label, while a box or points identify the object**. A text-only request explains which viewer hint is missing; it never silently calls a hosted localizer. Annotate one object/label per request, then refine or annotate another structure. An existing label is replaced only within the requested scope; overlapping other labels require correction.

### Radiology

#### Slicer

1. Open a sample and select **SAM 2.1** for a slice or **MedSAM2** for a volume.
2. Create slice boxes and positive/negative points through chat, or use native Markups. Chat-created hints remain editable; drag their handles to refine them. For a manually drawn box or point list, choose its node in **SAM hint**. A selected 3D ROI retains the existing volume-seed workflow.
3. Choose a source-aligned slice with a suitable intensity window. The assistant combines the target's current slice box and points, including native edits. Select the intended box if several match. A point label beginning with `-` or `negative` is an exclusion point.
4. Run SAM on the next chat turn after editing hints. Inspect the returned mask before submitting it.

#### OHIF

Open either NIfTI or imported DICOM. Chat creates editable native RectangleROI boxes and Probe points; green points include and red points exclude. Drag the handles to adjust them. With a SAM model selected, **Spatial prompt → Box / Point** also supports drawing manually. Native Probe labels starting with `-` or `negative` supply exclusions. One matching box and all matching points on the current slice are combined; select one box when several match.

#### Chat in both viewers

For example, on a source slice large enough to contain these coordinates:

```text
Create a box for spleen from voxel 50, 60 to voxel 120, 140 on this slice
Add a positive point in the box for spleen
Add a negative point for spleen at voxel 125, 145 on this slice
Move the positive point for spleen to voxel 90, 100 on this slice
Resize the spleen box to voxel 45, 55 through voxel 125, 145 on this slice
Segment spleen on this slice using SAM 2.1 with this box and these points
Clear the negative points on this slice
Clear the spleen box on this slice
Clear all SAM prompts on this slice
```

Coordinates are **zero-based source voxels**, not screen pixels. Two coordinates specify the two source axes other than the slice axis, in source-axis order; three specify I/J/K. Do not copy the example coordinates without checking your image. A point requested inside a box starts at its center and is explicitly described as an editable starting point, not anatomical localization. Without coordinates or a suitable box/selection, the assistant requests the missing information.

For model-assisted localization instead of explicit coordinates, name a compatible vision model:

```text
Create an initial box for spleen on this slice using GPT Sol
```

This is a separate localization job and may use a hosted API. SAM does not locate organs from names alone. Choose SAM for the subsequent segmentation request. No automatic paid-model fallback is used.

Clear commands distinguish target, point polarity, boxes versus points, and current slice versus full volume. For example, `Clear all SAM prompts for spleen throughout this volume` preserves other targets. Hints remain viewer-local; clearing them does not clear segmentation voxels or saved annotations. Save a Slicer scene to retain its hints; OHIF hints currently last for the viewer session. Chat hint edits do not enter the segmentation undo stack.

Pending edits and SAM proposals are rejected if their source slice, hint geometry/selection, sample or annotation revision becomes stale. Keep the seed slice selected until completion. Existing selected Slicer 3D ROIs retain the ROI capture path; chat edit commands operate on slice boxes and points.

For volume propagation use:

```text
Annotate spleen through the full volume using MedSAM2 and my box
```

Review the entire result before submission. A box is a prompt, not a guarantee that every enclosed voxel belongs to the named organ.

### Pathology and other 2D images

In QuPath, select **one object-sized region**, choose **SAM 2.1**, then say:

```text
Annotate the selected region as nucleus using SAM 2.1
```

The selected region is sent as one crop, without tiling, and its bounding box prompts the object. The returned mask is placed in source coordinates and restricted to the selection. This supports interactive object annotation; a broad region containing hundreds of nuclei needs a dedicated instance-segmentation model. SAM is not presented as a cell-type classifier or automatic whole-slide nuclei detector.

### Provenance and execution limits

| Model | Pinned checkpoint | Runtime |
| --- | --- | --- |
| SAM 2.1 tiny | [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5) | Image predictor |
| MedSAM2 | [wanglab/MedSAM2](https://huggingface.co/wanglab/MedSAM2/tree/e4a6f35edd7e091619cbc0750f462f1574e23955), `MedSAM2_latest.pt` | Medical image predictor and bidirectional volume propagation |

The SAM package uses the [MedSAM2 runtime pinned at 332f30d](https://github.com/bowang-lab/MedSAM2/tree/332f30d420f1d1b08e2a79b3ae6a602458808383), which shares the SAM 2 architecture. Its published wheel omits model YAML resources, so the two required configurations are packaged with this adapter; provenance and the upstream license are retained alongside them. Model loading uses `weights_only=True`. Frame preprocessing follows the published model interface; native source geometry is restored without resampling the original image.

SAM training recipes, streaming video, automatic whole-slide object discovery, and automatic evaluation without spatial prompts remain unimplemented. Evaluation must never derive prompts from held-out masks and call the result automatic annotation. The actual GPU smoke checks verify execution, source geometry and viewer transfer; they are not clinical accuracy benchmarks.

## DICOM import and viewing

### Connect, filter and import

Open **Datasets → Import from DICOM server**:

1. Enter the **DICOMweb endpoint**, such as `http://localhost:8042/dicom-web`. This address is reached by the MONAI Label backend, including when your browser is on a tablet.
2. Choose no authentication, username/password, or access token. Click **Connect**. Credentials are stored encrypted in the workspace and are never sent to the viewer. Saved connections can be reused within the project.
3. Click **Search**, with optional filters:

   | Filter | Matching |
   | --- | --- |
   | Modality | CT, MR or another listed modality |
   | Study date from/to | Inclusive range; either end can be omitted |
   | Import status | Not imported (default), already imported, or all |
   | Patient ID / accession number | Exact value |
   | Patient name / study or series description | Contains the entered text, subject to the source server's DICOM matching rules |
   | Study UID / series UID | Exact identifier, under More filters → Exact UID lookup |

4. Select series, or choose **Import all matches**. This includes available matches across result pages. Searches are capped at 1,000 series; refine filters if more match. Already imported and unsupported series are not imported again.

The activity job reports imported, skipped and failed series individually. Cancellation retains completed series; retry skips those already imported. A failed or incomplete series does not become a dataset asset.

The endpoint must provide standard QIDO `/series` search with pagination and WADO instance retrieval. A raw PACS/DIMSE address or Orthanc REST root is not a DICOMweb endpoint. Unsupported source filters or authentication are reported by the connection/search response.

### Where images are served

Import copies original DICOM instances into the project's workspace and creates a geometry-preserving scalar volume for annotation and training. **OHIF reads the imported workspace files through authenticated `/api/assets/{asset_id}/dicomweb` routes. The original server is not contacted during viewing.** Each annotation session shows only the selected sample's study and series, so copies in other projects do not appear as duplicate studies. Access is checked at project, series and instance level.

Original DICOM headers and UIDs are retained. Patient identity, when present, groups imported series; missing patient identity falls back to the study. Identical decoded images reuse their existing source group; conflicting groups require correction before import. Check grouping before dividing a dataset for learning, especially when the same patient appears under different archive identities.

### Open NIfTI in OHIF

Every 3D sample offers **Slicer** and **OHIF**. Select OHIF, or say `view this sample in OHIF`.

The dataset's **OHIF** action opens a new tab immediately and loads the viewer there once preparation completes. Keep the workspace tab open during setup. If a browser blocks the new tab, the assistant provides an **Open OHIF annotation viewer** link.

On first use, MONAI Label generates and caches a DICOM **Secondary Capture** viewing series locally. Later launches reuse it. No Orthanc upload or running DICOM server is needed. The original NIfTI and its source grid remain the annotation/training source; edits from OHIF map back to that grid.

The viewing copy uses synthetic identifiers and explicitly says it is derived from NIfTI. Its modality is OT because NIfTI does not establish the original acquisition modality or clinical headers. Integer intensities within signed 16-bit range are preserved exactly; other intensities use 16-bit quantization with a recorded rescale slope/intercept. This affects the viewing copy only.

### Limits

- Imported DICOM: regular scalar single-frame CT/MR, up to 3,000 instances and 256 MiB per series. Enhanced multiframe, irregular spacing, gantry tilt and unsupported decoders are rejected.
- NIfTI viewing: scalar 3D with orthogonal voxel axes, including oblique rotations and reversed slice direction. Sheared grids require resampling outside this preview or viewing in Slicer.
- Local DICOMweb is a read-only subset for the bundled OHIF viewer, not a full PACS or DICOM archive. DICOM SEG export remains future work.
- Original copies can contain patient-identifying metadata; workspace access follows the existing project roles.

The optional [local Orthanc example](../deploy/dicom/compose.yaml) is useful as an import source. It is independent of local viewing after import. The SeriesInstanceUID API/CLI import uses `MONAILABEL_ORTHANC_URL` for its import step.

Protocol references: [DICOM QIDO search](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_10.6.html), [DICOM WADO retrieval](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_10.4.html).

## Voice and remote access

The web workspace and OHIF have two independent controls:

- **Microphone:** tap the microphone inside the prompt box, speak, then tap the stop icon. Review or edit the transcript and press the send arrow. Speech never submits a prompt automatically.
- **Read replies:** opt in to spoken assistant responses. Turn it off to stop playback.

Recognition uses the device browser’s speech service. MONAI Label receives the resulting prompt text; it does not record or store microphone audio. Some browsers send audio to their speech provider, so this is not an offline transcription feature. Support varies by browser; unsupported browsers show an explanation and retain typing/keyboard dictation. See [MDN’s SpeechRecognition notes](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).

For Safari, enable Siri and allow microphone/speech access when prompted. WebKit’s speech recognition uses the Siri speech engine. See [WebKit’s support notes](https://webkit.org/blog/11648/new-webkit-features-in-safari-14-1/). Denied access, unavailable microphones and service connection errors appear beside the composer.

### Phones, tablets and other computers

Use **HTTPS** with a certificate trusted by the device when connecting over your network. `localhost` on a phone refers to the phone, not your workstation. Connect to the server’s network hostname or IP address. The same responsive workspace runs on touch devices; the Menu and Assistant buttons expose navigation and chat on small screens.

OHIF runs in the device browser. On tablets its study list starts collapsed; on phones both side panels start collapsed to leave room for the image. Tap the right-side panel icon to open the assistant. Swipe the phone toolbar to reach additional tools. Review controls are under **Review annotation**. Desktop Slicer and QuPath require a desktop installation and cannot launch on a phone or tablet through the server.

If you already use an HTTPS reverse proxy, keep it and include the hostname in `MONAILABEL_ALLOWED_HOSTS`. For a local demonstration, [mkcert](https://github.com/FiloSottile/mkcert) can create a trusted development certificate:

```bash
## Install mkcert using its platform instructions first.
mkcert -install
mkdir -p .local-certs
## Replace YOUR_SERVER_IP with the workstation's actual LAN address.
mkcert -cert-file .local-certs/server.pem -key-file .local-certs/server-key.pem \
  localhost 127.0.0.1 YOUR_SERVER_IP
```

Trust mkcert’s **rootCA.pem** on the phone/tablet using its OS instructions; `mkcert -CAROOT` shows the directory. Do not transfer the CA private key. The mkcert README covers mobile trust setup.

Stop the existing MONAI Label process before using the same workspace with these arguments:

```bash
MONAILABEL_ALLOWED_HOSTS=localhost,127.0.0.1,YOUR_SERVER_IP \
uv run monailabel-server --host 0.0.0.0 --port 8443 \
  --ssl-certfile .local-certs/server.pem --ssl-keyfile .local-certs/server-key.pem
```

Keep any annotation/coordinator key variables in that shell’s environment. Open `https://YOUR_SERVER_IP:8443` from a device on the same network, sign in, and allow the microphone. Open a DICOM sample in OHIF from Datasets for a browser-based annotation demo. Network/firewall access must allow the chosen port.

### Verification limits

Automated checks simulate speech events to verify transcript updates, editing, cancellation, permission errors and spoken-reply controls. Responsive layouts are inspected at desktop, tablet and phone sizes. Headless browsers do not establish real microphone recognition quality or physical iOS/Android compatibility; check those on the device you will use for a demo.
