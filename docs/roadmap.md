# Planned work

These capabilities are not complete. Current setup and workflows are in the [README](../README.md).

- **Video/endoscopy:** import clips and edit instrument tracks in an annotation-only CVAT flow. Core/backend groundwork exists; the complete import → editor → review loop is not available. CVAT will open directly into an annotation task; MONAI Label owns project and dataset management. The adapter must preserve frame timestamps, track identities, occlusion/outside states and revisions, with patient/procedure grouping. CVAT backend services are still required.
- **Pathology:** streaming whole slides, durable instance/cell-type annotations, dedicated local nuclei providers and their review/training recipes.
- **Editing:** range-limited segmentation, shared box/ROI undo, richer navigation and morphology prompts, and durable browser viewer drafts.
- **Learning:** SAM training, video tracking evaluation, arbitrary network/transform onboarding and windowed volume evaluation for slice-only vision models.
- **Interoperability:** DICOM SEG export, XNAT integration and remote desktop launch.
- **Deployment:** native Windows/macOS QA and installer completion, fresh managed Lightning startup verification, SSO and distributed workers.

Keep future features behind explicit capabilities; unavailable actions should not be presented as working. Preserve revisions, geometry, patient/procedure separation and independent reference data as these integrations are added.
