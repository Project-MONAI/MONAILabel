"use strict";

// Deletion uses explicit forms, never an assistant's direct destructive tool call.
function confirmationStyle() {
  const form = document.querySelector("#action-form");
  const submit = form.querySelector('[type="submit"]');
  const cancelButton = form.querySelector("[data-cancel-delete]");
  submit.className = "danger";
  const actions = document.createElement("div");
  actions.className = "form-actions";
  actions.append(cancelButton, submit);
  form.append(actions);
  cancelButton.addEventListener("click", () => {
    document.querySelector("#dialog").close();
  });
  return form;
}
const retention = `<p class="muted">Unused server file copies are reclaimed at the next server restart. Original files, external DICOM servers and shared model/viewer downloads are kept.</p>`;
const cancel = '<button type="button" data-cancel-delete>Cancel</button>';

export function deleteProject({
  project,
  api,
  modal,
  field,
  escapeHTML,
  finished,
}) {
  modal(
    "Delete project",
    `<p>Permanently delete <strong>${escapeHTML(project.name)}</strong> and its imported samples, annotations, reviews, snapshots, project models, credentials and chat history? This cannot be undone.</p>` +
      retention +
      field(
        "Type the project name to confirm",
        "confirmation_name",
        "text",
        "required autocomplete='off'",
      ) +
      cancel,
    async (form) => {
      await api(`/projects/${project.id}`, "DELETE", {
        confirmation_name: form.get("confirmation_name"),
      });
      await finished();
    },
    "Delete project",
  );
  const form = confirmationStyle();
  const input = form.elements.namedItem("confirmation_name");
  const submit = form.querySelector('[type="submit"]');
  submit.disabled = true;
  input.addEventListener("input", () => {
    submit.disabled = input.value !== project.name;
  });
  input.focus();
}

export function deleteFiles({
  project,
  assets,
  api,
  modal,
  escapeHTML,
  finished,
}) {
  if (!assets.length) throw new Error("Select the files to delete first.");
  modal(
    `Delete ${assets.length} file${assets.length === 1 ? "" : "s"}`,
    `<p>Delete these imported samples and their annotations, proposals and review history? This cannot be undone.</p><ul class="deletion-files">${assets.map((a) => `<li>${escapeHTML(a.name)}</li>`).join("")}</ul><p>Unused evaluation imports can be deleted. Files used in saved evaluation references, snapshots or trained models are protected.</p>` +
      retention +
      cancel,
    async () => {
      await api(`/projects/${project.id}/assets`, "DELETE", {
        asset_ids: assets.map((a) => a.id),
      });
      await finished();
    },
    "Delete files",
  );
  confirmationStyle();
}
