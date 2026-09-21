# Roadmap

Current setup and workflows are in the [README](../README.md).

## Implemented

- **Browser desktops:** Slicer and QuPath launch as native windows for local access and in browser tabs for remote access, with no client installation. Adaptive resolution, clipboard transfer and session cleanup on tab close are supported; exiting the native viewer also closes its tab or returns to the workspace. The server currently requires Linux Docker and uses software rendering. See [viewer setup](viewers.md#local-and-remote-launch).
- **Video/endoscopy:** managed CVAT installation, integrated chat, model-assisted box/polygon annotation, SAM 2.1 tracking over selected ranges or whole clips, immutable submission and workspace review. Docker and FFmpeg are required. See [video annotation](video.md).

## Remaining

- **Model connections:** unify configuration for NVIDIA gateway presets and user-added models, including endpoint and credential changes while preserving model provenance. For now, gateway presets retain their NVIDIA connection; use a separate model connection for another provider or account.
- **Video/endoscopy:** automatic rediscovery after tracking loss, native pixel-mask track editing, additional temporal trackers and external CVAT team account provisioning.
- **Pathology:** streaming whole slides, durable instance/cell-type annotations, dedicated local nuclei providers and their review/training recipes.
- **Editing:** volume slice-range segmentation, shared box/ROI undo, richer navigation and morphology prompts, and automatic recovery of unsaved viewer drafts.
- **Learning:** SAM training, video tracking evaluation, arbitrary network/transform onboarding and windowed volume evaluation for slice-only vision models.
- **Interoperability:** DICOM SEG export and XNAT integration.
- **Deployment:** native Windows/macOS QA and installer completion, physical tablet/browser compatibility checks, GPU-accelerated browser desktop rendering, fresh managed Lightning startup verification, SSO and distributed workers.

Keep future features behind explicit capabilities; unavailable actions should not be presented as working. Preserve revisions, geometry, patient/procedure separation and independent reference data as these integrations are added.
