import { escapeHTML as esc } from "./ui.js";
import { randomId } from "./random-id.js";

const stem = (name) =>
  name.toLowerCase().replace(/\.(nii\.gz|nii|png|tiff?|jpe?g|svs)$/, "");
const labelStem = (name) =>
  stem(name).replace(/[_-](mask|labels?|seg|segmentation)$/, "");

export function pairReferenceFiles(images, labels) {
  if (!images.length || !labels.length)
    throw new Error("Choose images and their reference label files.");
  if (images.length === 1 && labels.length === 1)
    return [[images[0], labels[0]]];
  const used = new Set();
  const pairs = images.map((image) => {
    const exact = labels.filter(
      (label) => stem(label.name) === stem(image.name),
    );
    const matching = exact.length
      ? exact
      : labels.filter((label) => labelStem(label.name) === stem(image.name));
    if (matching.length !== 1 || used.has(matching[0]))
      throw new Error(
        `Match one label file to ${image.name}. Use the same filename, optionally ending in _mask or _labels.`,
      );
    used.add(matching[0]);
    return [image, matching[0]];
  });
  if (used.size !== labels.length)
    throw new Error(
      "Some label files have no matching image. Check the selected files.",
    );
  return pairs;
}

export function upload(path, form, progress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api${path}`);
    request.timeout = 300000;
    if (form instanceof File)
      request.setRequestHeader("Content-Type", "application/octet-stream");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) progress(event.loaded / event.total);
    };
    request.onload = () => {
      try {
        const result = JSON.parse(request.responseText);
        if (request.status >= 200 && request.status < 300) resolve(result);
        else
          reject(
            new Error(
              typeof result.detail === "string"
                ? result.detail
                : "Import failed. Check the selected files and settings.",
            ),
          );
      } catch {
        reject(new Error(`Server returned HTTP ${request.status}.`));
      }
    };
    request.onerror = () =>
      reject(new Error("Connection lost. Retry the import."));
    request.ontimeout = () =>
      reject(new Error("Import timed out. Retry the import."));
    request.send(form);
  });
}

export function importFiles(ui, defaultUse = "pool") {
  const { state, api, modal, field, selectField, message } = ui;
  const projectId = state.project.id;
  const sets = state.evaluationSets.filter((s) => !s.archived);
  let currentSet = null;
  const imported = new Set();
  const relatedGroup = `case:${randomId()}`;
  const initialName = state.project.labels.filter((l) => l.id);
  const row = (value = 1, name = "") =>
    `<div class="reference-label-row"><label>Label value<input type="number" data-label-value min="1" max="65535" value="${value}" required></label><label>Structure<input type="text" data-label-name list="reference-structure-names" maxlength="80" value="${esc(name)}" placeholder="e.g. Spleen" required></label><button type="button" data-remove-structure aria-label="Remove structure" title="Remove structure">×</button></div>`;
  modal(
    "Import files",
    '<div class="dataset-import-choices">' +
      selectField(
        "Dataset use",
        "split",
        '<option value="pool">Annotation & training</option><option value="validation">Evaluation only</option>',
      ) +
      selectField(
        "What to import",
        "content",
        '<option value="images">Images only</option><option value="labels">Images + labels</option>',
      ) +
      '</div><p data-purpose-help class="muted"></p>' +
      field(
        "Image files",
        "images",
        "file",
        "required multiple accept='.nii,.nii.gz,.png,.jpg,.jpeg,.tif,.tiff,.svs'",
        "Up to 128 MiB per file.",
      ) +
      "<div data-label-options>" +
      field(
        "Label files",
        "labels",
        "file",
        "required multiple accept='.nii,.nii.gz,.png,.tif,.tiff'",
        "Match image filenames, for example case01.nii.gz and case01_mask.nii.gz.",
      ) +
      '<p data-pair-status role="status"></p>' +
      `<fieldset><legend>Structures in these labels</legend><p class="muted">0 is background. Name each structure covered by these label files.</p><datalist id="reference-structure-names">${initialName.map((l) => `<option value="${esc(l.name)}">`).join("")}</datalist><div data-label-rows>${row(1, initialName.length === 1 ? initialName[0].name : "")}</div><button type="button" data-add-structure>Add structure</button></fieldset>` +
      '<label class="reference-import-choice"><input type="checkbox" name="reviewed">These labels have been reviewed</label><p class="muted">Leave unchecked to review them after import.</p></div>' +
      "<div data-evaluation-options>" +
      selectField(
        "Evaluation set",
        "evaluation_set",
        '<option value="">Create a new set</option>' +
          sets
            .map((s) => `<option value="${s.id}">${esc(s.name)}</option>`)
            .join(""),
      ) +
      `<div data-set-name>${field("Set name", "set_name", "text", 'required maxlength="120" placeholder="e.g. External spleen references"')}</div></div>` +
      `<details><summary>Advanced options</summary><div data-label-options>${field("Label source (optional)", "source", "text", 'maxlength="500" placeholder="e.g. Expert-reviewed cohort"')}</div><label class="reference-import-choice"><input type="checkbox" name="related">All images are from the same patient or slide</label><div data-reference-group hidden>${field("Patient or slide ID (optional)", "group_id", "text", 'maxlength="200"', "Otherwise each image is a separate case.")}</div></details>` +
      '<div data-import-status hidden><p role="status"></p><progress max="1" value="0" aria-label="File import"></progress><ul></ul></div>',
    async (data) => {
      const evaluation = data.get("split") === "validation";
      const withLabels = evaluation || data.get("content") === "labels";
      const images = data
        .getAll("images")
        .filter((f) => f.size && !f.name.startsWith("._"));
      if (!images.length) throw new Error("Choose image files to import.");
      const pairs = withLabels
        ? pairReferenceFiles(
            images,
            data
              .getAll("labels")
              .filter((f) => f.size && !f.name.startsWith("._")),
          )
        : images.map((image) => [image, null]);
      const labels = {};
      if (withLabels) {
        document
          .querySelectorAll("[data-label-rows] .reference-label-row")
          .forEach((row) => {
            const value = Number(row.querySelector("[data-label-value]").value);
            const name = row.querySelector("[data-label-name]").value.trim();
            if (
              !name ||
              !Number.isInteger(value) ||
              value < 1 ||
              value > 65535 ||
              labels[value]
            )
              throw new Error(
                "Give each structure a unique label value and a name.",
              );
            labels[value] = name;
          });
        if (!Object.keys(labels).length)
          throw new Error("Add the structures covered by these labels.");
      }
      currentSet ||=
        sets.find((s) => s.id === data.get("evaluation_set")) || null;
      const group = data.has("related")
        ? data.get("group_id").trim() || relatedGroup
        : null;
      const host = document.querySelector("[data-import-status]");
      host.hidden = false;
      const status = host.querySelector('[role="status"]'),
        progress = host.querySelector("progress"),
        list = host.querySelector("ul");
      list.replaceChildren();
      let failed = 0,
        covered = null;
      const fields = [
        ...document.querySelectorAll("#action-form input, #action-form select"),
      ];
      const disabled = fields.map((input) => input.disabled);
      fields.forEach((input) => (input.disabled = true));
      try {
        for (const [index, [image, mask]] of pairs.entries()) {
          const item = document.createElement("li");
          list.append(item);
          status.textContent = `Importing ${index + 1} of ${pairs.length}: ${image.name}`;
          progress.value = 0;
          try {
            if (image.size > 128 * 1024 ** 2 || mask?.size > 128 * 1024 ** 2)
              throw new Error("Each file must be at most 128 MiB.");
            const progressChanged = (value) => {
              progress.value = value;
              item.textContent = `${image.name} — ${value < 1 ? `uploading ${Math.round(value * 100)}%` : "processing…"}`;
            };
            if (withLabels) {
              const metadata = {
                split: data.get("split"),
                labels,
                reviewed: data.has("reviewed"),
                source: (data.get("source") || "").trim(),
                group_id: group,
                ...(evaluation
                  ? currentSet
                    ? {
                        evaluation_set_id: currentSet.id,
                        base_version: currentSet.version,
                      }
                    : { evaluation_set_name: data.get("set_name").trim() }
                  : {}),
              };
              const body = new FormData();
              body.set("metadata", JSON.stringify(metadata));
              body.set("image", image);
              body.set("labels", mask);
              const result = await upload(
                `/projects/${projectId}/label-imports`,
                body,
                progressChanged,
              );
              currentSet = result.evaluation_set;
              covered = result.covered_labels;
            } else {
              const params = new URLSearchParams({
                name: image.name,
                split: data.get("split"),
                shared: "true",
              });
              if (group) params.set("group_id", group);
              const identity = `${params}:${image.size}:${image.lastModified}`;
              if (!imported.has(identity)) {
                await upload(
                  `/projects/${projectId}/assets/upload?${params}`,
                  image,
                  progressChanged,
                );
                imported.add(identity);
              }
            }
            item.textContent = `${image.name} — imported`;
          } catch (error) {
            failed++;
            item.className = "form-error";
            item.textContent = `${image.name} — ${error.message}`;
          }
        }
        if (failed) {
          status.textContent = `${pairs.length - failed} imported · ${failed} failed. Successful imports are saved; retry to finish.`;
          return false;
        }
        if (evaluation && data.has("reviewed")) {
          status.textContent = "Saving fixed evaluation references…";
          await api(
            `/projects/${projectId}/evaluation-sets/${currentSet.id}/versions`,
            "POST",
            { base_version: currentSet.version, label_ids: covered },
          );
        }
        state.page = "datasets";
        state.datasetFilter = evaluation ? `set:${currentSet.id}` : "all";
        state.pages.datasets = 1;
        message(
          `Imported ${pairs.length} ${withLabels ? "image/label pairs" : "images"}${evaluation ? ` into ${currentSet.name}. Reserved for evaluation. ${data.has("reviewed") ? "Fixed references are ready for model comparison." : "Review the labels before comparing models."}` : "."}`,
        );
      } finally {
        fields.forEach((input, index) => (input.disabled = disabled[index]));
      }
    },
    "Import files",
  );
  const form = document.querySelector("#action-form");
  form.elements.split.value = defaultUse;
  const sync = () => {
    const evaluation = form.elements.split.value === "validation";
    if (evaluation) form.elements.content.value = "labels";
    form.elements.content.disabled = evaluation;
    const withLabels = form.elements.content.value === "labels";
    for (const [selector, visible] of [
      ["[data-label-options]", withLabels],
      ["[data-evaluation-options]", evaluation],
    ]) {
      for (const panel of form.querySelectorAll(selector)) {
        panel.hidden = !visible;
        panel
          .querySelectorAll("input, select, button")
          .forEach((input) => (input.disabled = !visible));
      }
    }
    const creating = evaluation && !form.elements.evaluation_set.value;
    form.querySelector("[data-set-name]").hidden = !creating;
    form.elements.set_name.disabled = !creating;
    form.querySelector("[data-purpose-help]").textContent = evaluation
      ? "Evaluation-only data needs images + labels and is excluded from every model’s training."
      : withLabels
        ? "Add images and existing labels. Each model manages its own split."
        : "Add images now; each model chooses its own training and validation split.";
  };
  form.elements.split.onchange = sync;
  form.elements.content.onchange = sync;
  form.elements.evaluation_set.onchange = () => {
    currentSet = null;
    sync();
  };
  form.elements.set_name.oninput = () => {
    currentSet = null;
  };
  const preview = () => {
    const images = [...form.elements.images.files].filter(
        (f) => !f.name.startsWith("._"),
      ),
      labels = [...form.elements.labels.files].filter(
        (f) => !f.name.startsWith("._"),
      );
    try {
      form.querySelector("[data-pair-status]").textContent =
        `${pairReferenceFiles(images, labels).length} image/label pairs matched.`;
    } catch (error) {
      form.querySelector("[data-pair-status]").textContent =
        images.length || labels.length ? error.message : "";
    }
  };
  form.elements.related.onchange = () => {
    form.querySelector("[data-reference-group]").hidden =
      !form.elements.related.checked;
  };
  form.elements.images.onchange = preview;
  form.elements.labels.onchange = preview;
  form.querySelector("[data-add-structure]").onclick = () => {
    const values = [...form.querySelectorAll("[data-label-value]")].map((i) =>
      Number(i.value),
    );
    form
      .querySelector("[data-label-rows]")
      .insertAdjacentHTML("beforeend", row(Math.max(0, ...values) + 1));
  };
  form.querySelector("[data-label-rows]").onclick = (event) => {
    event.target
      .closest("[data-remove-structure]")
      ?.closest(".reference-label-row")
      ?.remove();
  };
  sync();
}
