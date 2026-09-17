import { paginationButton } from "./icons.js";
import { escapeHTML as esc } from "./ui.js";

export function trainingSamples() {
  return `<details data-sample-filters><summary>Filter samples</summary><p class="muted" data-sample-filter-summary>All reviewed and accepted samples.</p><label>Annotation status<select name="training_labels"><option value="reviewed">Reviewed and accepted</option><option value="predictions">Include unreviewed predictions (experimental)</option></select></label><div id="training-note" hidden><p class="muted">Predictions may contain mistakes. Evaluation still requires accepted annotations.</p><label>Training note (optional)<input name="training_note" maxlength="1000"></label></div><label>Images<select name="training_scope"><option value="all">All matching images</option><option value="selected">Selected images</option></select></label><div data-training-selection hidden><label>Find images<input type="search" data-training-search placeholder="Search image names"></label><div class="training-sample-picker" data-training-images></div></div><label>Sample limit<input name="training_limit" type="number" min="1" max="10000" step="1" placeholder="All matching images"></label><small>Filters apply to training only. Evaluation images stay separate; limits keep related images together.</small><div hidden data-training-values></div></details>`;
}

export function bindTrainingSamples(form, { state, learner, latestDecision }) {
  const host = form.querySelector("[data-sample-filters]");
  const record = state.modelSplits.find((s) => s.learner_id === learner.id);
  const selected = new Set(state.selectedFiles);
  const imageList = host.querySelector("[data-training-images]");
  const annotated = state.assets.filter((a) => a.annotation_id);
  const pageSize = 10;
  let page = 0;
  let visible = [];
  const eligibleAssets = () =>
    annotated.filter((asset) => {
      const verdict = latestDecision(asset)?.verdict || "pending";
      return (
        asset.split !== "validation" &&
        !record?.validation_groups.includes(asset.group_id) &&
        !record?.validation_image_keys.includes(asset.image_key) &&
        (verdict === "accepted" ||
          (form.elements.training_labels.value === "predictions" &&
            verdict === "pending"))
      );
    });
  const update = () => {
    const predictions = form.elements.training_labels.value === "predictions";
    const picking = form.elements.training_scope.value === "selected";
    host.querySelector("[data-training-selection]").hidden = !picking;
    host.querySelector("#training-note").hidden = !predictions;
    const search = host
      .querySelector("[data-training-search]")
      .value.trim()
      .toLowerCase();
    const eligible = eligibleAssets();
    const chosen = eligible.filter((asset) => selected.has(asset.id));
    const matches = eligible.filter((asset) =>
      asset.name.toLowerCase().includes(search),
    );
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.max(0, Math.min(page, pages - 1));
    visible = matches.slice(page * pageSize, (page + 1) * pageSize);
    const pageSelected = visible.filter((asset) =>
      selected.has(asset.id),
    ).length;
    imageList.innerHTML = `<div class="training-selection-heading"><span role="status">${chosen.length} selected</span><button type="button" class="model-text-action" data-training-clear ${chosen.length ? "" : "disabled"}>Clear selection</button></div>
      <div class="table-wrap"><table class="training-image-table" aria-label="Training images"><thead><tr><th scope="col" class="check-cell"><input type="checkbox" data-training-all aria-label="Select all images on this page" ${visible.length ? "" : "disabled"} ${visible.length && pageSelected === visible.length ? "checked" : ""}></th><th scope="col">Image</th><th scope="col">Status</th></tr></thead><tbody>${visible.map((asset) => `<tr class="${selected.has(asset.id) ? "row-selected" : ""}"><td class="check-cell"><input type="checkbox" data-training-image="${esc(asset.id)}" aria-label="${esc(asset.name)}" ${selected.has(asset.id) ? "checked" : ""}></td><td class="sample-cell">${esc(asset.name)}</td><td>${latestDecision(asset)?.verdict === "accepted" ? "Accepted" : "Unreviewed"}</td></tr>`).join("") || '<tr><td colspan="3">No matching images</td></tr>'}</tbody></table></div>
      <div class="pagination"><span>${matches.length ? `${page * pageSize + 1}–${Math.min((page + 1) * pageSize, matches.length)} of ${matches.length}` : "0 images"}</span><div>${paginationButton("left", `data-training-page="-1" ${page === 0 ? "disabled" : ""}`)}<span>Page ${page + 1} of ${pages}</span>${paginationButton("right", `data-training-page="1" ${page + 1 === pages ? "disabled" : ""}`)}</div></div>`;
    imageList.querySelector("[data-training-all]").indeterminate =
      pageSelected > 0 && pageSelected < visible.length;
    // Only eligible selections enter the request, including choices on other pages.
    host.querySelector("[data-training-values]").innerHTML = picking
      ? chosen
          .map(
            (asset) =>
              `<input type="hidden" name="training_assets" value="${esc(asset.id)}">`,
          )
          .join("")
      : "";
    const summary = [
      predictions ? "Includes unreviewed predictions" : "Reviewed and accepted",
    ];
    if (picking) summary.push(`${chosen.length} selected images`);
    if (form.elements.training_limit.value)
      summary.push(`up to ${form.elements.training_limit.value} images`);
    host.querySelector("[data-sample-filter-summary]").textContent =
      summary.join(" · ");
  };
  form.addEventListener("change", (event) => {
    const input = event.target;
    const assetId = input.dataset.trainingImage;
    if (assetId) {
      if (input.checked) selected.add(assetId);
      else selected.delete(assetId);
    } else if (input.hasAttribute("data-training-all")) {
      for (const asset of visible) {
        if (input.checked) selected.add(asset.id);
        else selected.delete(asset.id);
      }
    } else if (["training_scope", "training_labels"].includes(input.name))
      page = 0;
    update();
    if (assetId)
      imageList
        .querySelector(`[data-training-image="${CSS.escape(assetId)}"]`)
        ?.focus({ preventScroll: true });
    else if (input.hasAttribute("data-training-all"))
      imageList
        .querySelector("[data-training-all]")
        .focus({ preventScroll: true });
  });
  imageList.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.hasAttribute("data-training-clear")) selected.clear();
    else if (button.hasAttribute("data-training-page"))
      page += Number(button.dataset.trainingPage);
    else return;
    update();
  });
  const search = host.querySelector("[data-training-search]");
  search.oninput = () => {
    page = 0;
    update();
  };
  search.onkeydown = (event) => {
    if (event.key === "Enter") event.preventDefault();
  };
  form.elements.training_limit.oninput = update;
  update();
}

export function trainingSampleRequest(form) {
  const sample_filter = {};
  if (form.get("training_scope") === "selected") {
    sample_filter.asset_ids = form.getAll("training_assets");
    if (!sample_filter.asset_ids.length)
      throw new Error("Select at least one training image.");
  }
  const rawLimit = form.get("training_limit");
  if (rawLimit) {
    const limit = Number(rawLimit);
    if (!Number.isInteger(limit) || limit < 1 || limit > 10000)
      throw new Error("Use a whole-number sample limit from 1 to 10,000.");
    sample_filter.limit = limit;
  }
  return {
    sample_filter,
    allow_unreviewed_training: form.get("training_labels") === "predictions",
    note: form.get("training_note") || "",
  };
}
