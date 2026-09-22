# Roadmap

Current setup and workflows are in the [README](../README.md).

## Implemented

- **Hosted models:** key-based discovery and named project imports for NVIDIA, OpenAI, Anthropic and Gemini; three fixed presets with automatic provider selection and project-specific provider/route switching. See [provider setup](providers.md#bring-your-own-vision-model).
- **Browser desktops:** Slicer and QuPath launch as native windows for local access and in browser tabs for remote access, with no client installation. Adaptive resolution, clipboard transfer and session cleanup on tab close are supported; exiting the native viewer also closes its tab or returns to the workspace. The server currently requires Linux Docker and uses software rendering. See [viewer setup](viewers.md#local-and-remote-launch).
- **Video/endoscopy:** managed CVAT installation, integrated chat, model-assisted box/polygon annotation, SAM 2.1 tracking over selected ranges or whole clips, immutable submission and workspace review. Docker and FFmpeg are required. See [video annotation](video.md).

- **Complete learning workflows:** independently review pathology regions and video frame ranges, train a new U-Net from accepted coverage in all three specialties, and use the trained model for subsequent segmentation. Preserve source grouping and exclude unreviewed pixels. Native CVAT and QuPath tests cover submission, review, local training and checkpoint reuse.
- **Optional evaluation:** train approved samples without a separate evaluation dataset, or reserve a requested percentage of independent source groups. Runs without evaluation report no held-out score. Browser and backend checks cover training without evaluation and grouped validation.

## In progress

- **Coordinator reliability:** the radiology/pathology learning stories and endoscopy prompts are exercised with Nemotron 3.5 Lightning and Nano 4B. Extend checks for long conversations and ambiguous viewer commands; Nano 4B remains experimental. Nano 9B is outside the current verification scope.

## Remaining

- **DGX Spark:** native QuPath/CVAT provisioning, OHIF, managed Lightning and local VISTA3D/SAM/U-Net execution are covered in the [Spark guide](spark.md). Slicer requires an explicit native source build or compatible installation. Extend device coverage and model-quality validation beyond the documented software checks.

- **Video/endoscopy:** automatic rediscovery after tracking loss, native pixel-mask track editing, additional temporal trackers and external CVAT team account provisioning.
- **Pathology:** streaming whole slides, durable instance/cell-type annotations, dedicated local nuclei providers and their review/training recipes.
- **Editing:** volume slice-range segmentation, shared box/ROI undo, richer navigation and morphology prompts, and automatic recovery of unsaved viewer drafts.
- **Learning:** SAM training, video tracking evaluation, arbitrary network/transform onboarding and windowed volume evaluation for slice-only vision models.
- **Interoperability:** DICOM SEG export and XNAT integration.
- **Deployment:** physical tablet/browser compatibility checks, GPU-accelerated browser desktop rendering, SSO and distributed workers.

## Deferred platforms

Native Windows/macOS installation and viewer validation are outside the current release focus. Prioritize Linux workstations and DGX Spark servers; remote clients continue to use a browser.

Keep future features behind explicit capabilities; unavailable actions should not be presented as working. Preserve revisions, geometry, patient/procedure separation and independent reference data as these integrations are added.
