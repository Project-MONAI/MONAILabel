# Roadmap

The following work is not yet complete:

| Area | Remaining work |
| --- | --- |
| Coordinator | More long-conversation and ambiguous-command checks; Nano 4B/9B remain experimental |
| Pathology | Streaming whole slides, durable cell-type annotations, dedicated nuclei providers and training recipes |
| Video | Recovery after tracking loss, pixel-mask editing, additional trackers, tracking evaluation, external CVAT account provisioning |
| Editing | Volume slice ranges, shared box/ROI undo, additional navigation/morphology commands, unsaved-draft recovery |
| Interactive segmentation | Scribble/lasso inputs, volume boxes and reusable inference sessions between updates |
| Learning | SAM and nnInteractive fine-tuning, arbitrary network/transform onboarding, windowed volume evaluation with slice-only models |
| Interoperability | DICOM SEG export, XNAT integration |
| Deployment | SSO, distributed workers, hardware-accelerated browser desktops, physical tablet/browser validation |
| DGX Spark | Broader device/model validation and Slicer extension coverage; [Slicer currently requires an explicit native build](spark.md#native-slicer) |

Native Windows/macOS installation and viewer validation are deferred. The release focus is Linux workstations and DGX Spark servers with browser clients.
