import { escapeHTML as esc } from "./ui.js";

export async function importDatasetTemplate(ui) {
  const { state, api, modal, field, selectField, watch, safely } = ui;
  const projectId = state.project.id;
  const catalog = await api(`/projects/${projectId}/dataset-templates`);
  modal(
    "Import sample dataset",
    '<div class="dataset-import-choices">' +
      selectField(
        "From where to import",
        "template_id",
        ["Radiology", "Pathology", "Video"]
          .map(
            (category) =>
              `<optgroup label="${category}">${catalog
                .filter((t) => t.category === category)
                .map((t) => `<option value="${t.id}">${esc(t.name)}</option>`)
                .join("")}</optgroup>`,
          )
          .join(""),
      ) +
      '<div id="dataset-content-choice"></div>' +
      selectField(
        "Dataset use",
        "split",
        '<option value="pool">Annotation & training</option><option value="validation">Evaluation only</option>',
      ) +
      '</div><div id="dataset-template-options"></div>',
    async (form) => {
      const template = catalog.find((t) => t.id === form.get("template_id"));
      if (!template.importable)
        throw new Error("Use the dataset website to download this collection.");
      const isVideo = template.kind === "video";
      const evaluation = !isVideo && form.get("split") === "validation";
      const includeMasks = evaluation || form.get("include_masks") === "yes";
      const allSamples = form.get("amount") === "all";
      const targets = includeMasks ? form.getAll("targets") : [];
      if (includeMasks && template.targets.length && !targets.length)
        throw new Error("Choose the structures to import.");
      if (targets.length > 31)
        throw new Error("Choose up to 31 structures for one project.");
      const job = await api(`/projects/${projectId}/dataset-imports`, "POST", {
        template_id: template.id,
        include_masks: includeMasks,
        targets,
        split: isVideo ? "pool" : form.get("split"),
        evaluation_set_id: evaluation
          ? form.get("evaluation_set") || null
          : null,
        section:
          !evaluation && form.get("include_masks") === "test"
            ? "test"
            : "training",
        offset: allSamples ? 0 : Number(form.get("skip") || 0),
        limit: allSamples ? null : Number(form.get("limit") || 1),
        channel: Number(form.get("channel") || 0),
      });
      state.page = "activity";
      safely(() => watch(job.id, projectId));
    },
    "Import samples",
  );
  const form = document.querySelector("#action-form");
  function renderOptions() {
    const template = catalog.find(
      (t) => t.id === form.elements.template_id.value,
    );
    const size =
      template.download_bytes >= 1e9
        ? `${(template.download_bytes / 1e9).toFixed(1)} GB`
        : `${(template.download_bytes / 1e6).toFixed(1)} MB`;
    const isVideo = template.kind === "video";
    form.elements.split.disabled = isVideo;
    form.elements.split.options[0].textContent = isVideo
      ? "Annotation only"
      : "Annotation & training";
    if (isVideo) {
      form.elements.split.value = "pool";
      form.elements.split.onchange = null;
      document.querySelector("#dataset-content-choice").innerHTML = selectField(
        "What to import",
        "include_masks",
        '<option value="no">Video clip only</option>',
      );
      document.querySelector("#dataset-template-options").innerHTML =
        `<p>${esc(template.description)}</p><p><a href="${esc(template.source_url)}" target="_blank" rel="noopener">Dataset website ↗</a> · ${esc(template.license)}</p>` +
        `<p class="muted">${template.cached ? "Already downloaded." : `Downloads ${size}.`}</p>` +
        '<p id="dataset-import-summary" role="status">Imports one clip into Datasets. Open CVAT to draw and save instrument tracks, then submit them for review.</p>' +
        '<p class="muted">No reference tracks are included. Accepted polygon annotations can train a segmentation model; boxes alone cannot. Repeating the import preserves your annotations and CVAT drafts.</p>';
      form.querySelector('[type="submit"]').disabled = !template.importable;
      return;
    }
    const isArchive = template.id !== "openslide-cmu-small";
    const hasTestImages = template.sections.includes("test");
    const normalize = (name) => name.toLowerCase().replaceAll("_", " ");
    const projectTargets = template.targets.filter((target) =>
      state.project.labels.some(
        (label) => label.id && normalize(label.name) === normalize(target),
      ),
    );
    const defaultTargets = projectTargets.length
      ? projectTargets.slice(0, 31)
      : template.targets.filter((target) =>
          ["liver", "spleen"].includes(target),
        );
    document.querySelector("#dataset-content-choice").innerHTML = selectField(
      "What to import",
      "include_masks",
      template.importable
        ? `<option value="no">Images only</option>` +
            (template.has_masks
              ? `<option value="yes">Images + labels</option>`
              : "") +
            (hasTestImages
              ? '<option value="test">Test images only (no labels)</option>'
              : "")
        : "<option>Download from dataset website</option>",
    );
    form.elements.include_masks.disabled = !template.importable;
    document.querySelector("#dataset-template-options").innerHTML =
      `<p>${esc(template.description)}</p><p><a href="${esc(template.source_url)}" target="_blank" rel="noopener">Dataset website ↗</a> · ${esc(template.license)}</p>` +
      (template.importable
        ? `<p class="muted">${template.cached ? "Already downloaded." : `Downloads ${size}${isArchive ? ", even for a few samples" : ""}.`}</p>` +
          (isArchive
            ? selectField(
                "How much to import",
                "amount",
                '<option value="sample">Quick start · 5 samples</option><option value="all">All samples</option>',
              )
            : "") +
          '<div id="template-evaluation-set" hidden>' +
          selectField(
            "Evaluation set",
            "evaluation_set",
            '<option value="">' +
              esc(template.name) +
              " evaluation (automatic)</option>" +
              state.evaluationSets
                .filter((s) => !s.archived)
                .map((s) => `<option value="${s.id}">${esc(s.name)}</option>`)
                .join(""),
          ) +
          '</div><p id="dataset-import-summary" role="status"></p><p id="dataset-mask-summary" class="muted" hidden></p>' +
          '<details id="dataset-advanced"><summary>Advanced options</summary>' +
          (isArchive
            ? '<div class="form-grid">' +
              field(
                "Number of samples",
                "limit",
                "number",
                'required min="1" max="2000" value="5"',
              ) +
              field(
                "Skip this many samples",
                "skip",
                "number",
                'required min="0" max="10000" value="0"',
              ) +
              '</div><p class="muted">Imports a subset, in filename order. To try the next 5 samples, skip the first 5.</p>'
            : "") +
          (template.channels.length
            ? selectField(
                "Image type (modality)",
                "channel",
                template.channels
                  .map((c, i) => `<option value="${i}">${esc(c)}</option>`)
                  .join(""),
              ) +
              "<small>Imports one scalar modality. Modalities from the same case stay in one source group.</small>"
            : "") +
          (template.targets.length
            ? `<fieldset id="dataset-targets" hidden><legend>Labels for (up to 31 structures)</legend><input type="search" id="dataset-target-filter" aria-label="Find a structure" placeholder="Find a structure…"><div class="dataset-target-list">${template.targets.map((t) => `<label><input type="checkbox" name="targets" value="${esc(t)}" ${defaultTargets.includes(t) ? "checked" : ""}>${esc(t.replaceAll("_", " "))}</label>`).join("")}</div></fieldset>`
            : "") +
          '<p class="muted">Existing annotations are kept when you repeat an import. Keep samples from the same patient together when separating training and evaluation data.</p></details>'
        : '<p class="muted">This collection is available from its authors. Automatic import is not yet supported here. Download supported images, then use Import files; whole slides can be opened in QuPath.</p>');
    form.querySelector('[type="submit"]').disabled = !template.importable;
    if (template.importable) {
      const sync = () => {
        const masks = form.elements.include_masks;
        const evaluation = form.elements.split.value === "validation";
        if (evaluation && template.has_masks) masks.value = "yes";
        masks.disabled = evaluation;
        form.querySelector('[type="submit"]').disabled =
          evaluation && !template.has_masks;
        document.querySelector("#template-evaluation-set").hidden =
          !evaluation || !template.has_masks;
        form.elements.evaluation_set.disabled = !evaluation;

        const targets = document.querySelector("#dataset-targets");
        if (targets) targets.hidden = masks.value !== "yes";
        const count = Number(form.elements.limit?.value || 1);
        const skip = Number(form.elements.skip?.value || 0);
        const allSamples = form.elements.amount?.value === "all";
        if (form.elements.amount) {
          form.elements.amount.options[0].textContent =
            count === 5 && skip === 0
              ? "Quick start · 5 samples"
              : `Custom subset · ${count} samples`;
          form.elements.limit.disabled = allSamples;
          form.elements.skip.disabled = allSamples;
        }
        const section = hasTestImages
          ? masks.value === "test"
            ? "test images"
            : "images"
          : count === 1 && !allSamples
            ? "sample"
            : "samples";
        const amount = allSamples
          ? "all"
          : `${isArchive ? "up to " : ""}${count}`;
        const assigned = {
          pool: "",
          train: " Added to your project's training set.",
          validation: " Reserved for evaluation and excluded from training.",
        }[form.elements.split.value];
        document.querySelector("#dataset-import-summary").textContent =
          evaluation && !template.has_masks
            ? "This dataset has no labels. Choose a labeled dataset for evaluation."
            : `Imports ${amount} ${section}${masks.value === "yes" ? " + labels" : ""}${!allSamples && skip ? `, skipping the first ${skip}` : ""}.${assigned}`;
        const maskSummary = document.querySelector("#dataset-mask-summary");
        maskSummary.hidden = masks.value !== "yes";
        const selected = [
          ...form.querySelectorAll('[name="targets"]:checked'),
        ].map((input) => input.value.replaceAll("_", " "));
        const names =
          selected.length <= 3
            ? selected.join(", ")
            : `${selected.length} structures`;
        maskSummary.textContent =
          (targets
            ? `Masks: ${names || "none selected"}. Change structures in Advanced options. `
            : "") +
          "These are the dataset’s existing labels. Review them before training or evaluation.";
      };
      form.elements.include_masks.onchange = sync;
      if (form.elements.amount) form.elements.amount.onchange = sync;
      form.elements.split.onchange = sync;
      for (const input of form.querySelectorAll(
        '[name="limit"], [name="skip"], [name="targets"]',
      ))
        input.oninput = sync;
      const filter = document.querySelector("#dataset-target-filter");
      if (filter)
        filter.oninput = () => {
          for (const row of form.querySelectorAll(".dataset-target-list label"))
            row.hidden = !row.textContent
              .toLowerCase()
              .includes(filter.value.toLowerCase());
        };
      sync();
    }
  }
  form.elements.template_id.value = "Task09_Spleen";
  form.elements.template_id.onchange = renderOptions;
  renderOptions();
}
