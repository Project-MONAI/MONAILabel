# Planned work

These capabilities are not complete. Current setup and workflows are in the [README](../README.md).

- **Video/endoscopy:** automatic rediscovery after tracking loss, native pixel-mask track editing, additional temporal trackers and external CVAT team account provisioning. Managed CVAT installation, integrated chat, model-assisted box/polygon annotation, SAM 2.1 tracking over selected ranges or whole clips, immutable submission and workspace review are [implemented](video.md); Docker and FFmpeg remain required.
- **Pathology:** streaming whole slides, durable instance/cell-type annotations, dedicated local nuclei providers and their review/training recipes.
- **Editing:** range-limited segmentation, shared box/ROI undo, richer navigation and morphology prompts, and durable browser viewer drafts.
- **Learning:** SAM training, video tracking evaluation, arbitrary network/transform onboarding and windowed volume evaluation for slice-only vision models.
- **Interoperability:** DICOM SEG export, XNAT integration and remote desktop launch.
- **Deployment:** native Windows/macOS QA and installer completion, fresh managed Lightning startup verification, SSO and distributed workers.

Keep future features behind explicit capabilities; unavailable actions should not be presented as working. Preserve revisions, geometry, patient/procedure separation and independent reference data as these integrations are added.
