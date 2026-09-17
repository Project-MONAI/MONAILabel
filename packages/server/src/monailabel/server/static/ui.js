import { actionIcons, actionLabel } from "./icons.js";
export const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );

const statusLabels = {
  unannotated: "Not submitted",
  pending: "Pending review",
  accepted: "Accepted",
  changes_requested: "Needs changes",
  pool: "Unassigned",
  train: "Training",
  validation: "Evaluation",
  succeeded: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
  queued: "Queued",
  running: "Running",
};
export const targetNames = (project, ids) =>
  project.labels
    .filter((label) => label.id && ids.includes(label.id))
    .map((label) => label.name)
    .join(", ");
export const jobName = (kind) =>
  ({
    annotation: "Annotation",
    annotate: "Annotation",
    batch_annotate: "Batch segmentation",
    training: "Training",
    train: "Training",
    training_report: "Training evaluation",
    evaluation: "Evaluation",
    evaluate: "Evaluation",
    viewer: "Open viewer",
    dicom_import: "DICOM import",
    dataset_import: "Dataset import",
    bounding_box: "Bounding box",
    roi: "Region of interest",
    classification: "Cell classification",
    classify: "Cell classification",
    select: "Case selection",
    selection: "Case selection",
  })[kind] || kind.replaceAll("_", " ");
export const jobFinished = (job) =>
  ["succeeded", "failed", "cancelled", "interrupted"].includes(job.status);

export const badge = (value) =>
  `<span class="badge ${escapeHTML(value)}">${escapeHTML(statusLabels[value] || value.replaceAll("_", " "))}</span>`;
export const button = (text, action, id = "", cls = "secondary") =>
  `<button type="button" class="${cls}" data-action="${action}" data-id="${escapeHTML(id)}">${actionLabel(text, actionIcons[action])}</button>`;
