import { paginationButton } from "./icons.js";
import { escapeHTML, button, targetNames } from "./ui.js";

export function availableTargets(state, model) {
  if (["openai-polygons", "openai-chat-polygons"].includes(model.provider))
    return null;
  if (
    model.provider === "vista3d" &&
    (model.read_only || model.inherit_targets)
  ) {
    return state.recipes.find((recipe) => recipe.id === model.provider)
      ?.supported_targets;
  }
  return state.project.labels
    .filter((label) => label.id && model.label_ids.includes(label.id))
    .map((label) => label.name);
}

export function targetSummary(state, model) {
  if (["sam2", "medsam2"].includes(model.provider))
    return (
      '<p class="muted">Name a target and select its box or points in the viewer. One object per request; labels are not restricted to a fixed organ list. MedSAM2 propagates a seed through a medical volume. Inference only; fine-tuning is not integrated.</p><p><a href="https://github.com/' +
      (model.provider === "sam2"
        ? "facebookresearch/sam2"
        : "bowang-lab/MedSAM2") +
      '" target="_blank" rel="noopener noreferrer">Original model documentation ↗</a></p>'
    );

  const targets = availableTargets(state, model);
  if (targets === null)
    return '<p class="muted">Targets are defined in your annotation prompt.</p>';
  if (!targets)
    return '<p class="muted">Structure list is loading. Refresh to try again.</p>';
  const source = state.recipes.find(
    (recipe) => recipe.id === model.provider,
  )?.documentation_url;
  return (
    button(
      `Supported structures · ${targets.length}`,
      "model-targets",
      model.id,
      "subtle",
    ) +
    (source
      ? `<p><a href="${escapeHTML(source)}" target="_blank" rel="noopener noreferrer">Original model documentation ↗</a></p>`
      : "") +
    (model.state_key
      ? `<p class="muted">Trained labels: ${escapeHTML(targetNames(state.project, model.label_ids))}</p>`
      : "")
  );
}

export function showModelTargets(state, id, modal) {
  const model = state.models.find((item) => item.id === id);
  const classIds =
    state.recipes.find((recipe) => recipe.id === model.provider)
      ?.target_class_ids || {};
  const hasClassIds = Object.keys(classIds).length > 0;
  const targets = [...(availableTargets(state, model) || [])].sort((a, b) =>
    a.localeCompare(b),
  );
  modal(
    `${model.name} structures`,
    `<p class="muted">${model.provider === "vista3d" && (model.read_only || model.inherit_targets) ? "The original bundle’s 117 default segmentation structures." : "Structures supported by this annotation model."}</p>
    <label>Find a structure<input id="target-search" type="search" placeholder="${hasClassIds ? "Search by name or class ID…" : "Search organs or structures…"}" autocomplete="off"></label>
    <div id="target-results"></div>`,
    () => {},
    "Done",
  );
  const search = document.querySelector("#target-search");
  const results = document.querySelector("#target-results");
  const pageSize = 8;
  let page = 0;
  const render = () => {
    const query = search.value.trim().toLowerCase();
    const searchById = hasClassIds && /^\d+$/.test(query);
    const matches = targets.filter((name) =>
      searchById
        ? classIds[name.toLowerCase()] === Number(query)
        : name.toLowerCase().includes(query),
    );
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.min(page, pages - 1);
    results.innerHTML = `<p class="muted" role="status">${matches.length} of ${targets.length} structures</p>
      <div class="target-catalog"><table aria-label="Supported structures"><thead><tr><th scope="col">Structure</th>${hasClassIds ? '<th scope="col" class="target-class-id">Class ID</th>' : ""}</tr></thead><tbody>${
        matches
          .slice(page * pageSize, page * pageSize + pageSize)
          .map(
            (name) =>
              `<tr><td>${escapeHTML(name[0].toUpperCase() + name.slice(1))}</td>${
                hasClassIds
                  ? `<td class="target-class-id">${escapeHTML(classIds[name.toLowerCase()] ?? "—")}</td>`
                  : ""
              }</tr>`,
          )
          .join("") ||
        `<tr><td colspan="${hasClassIds ? 2 : 1}">No matching structures</td></tr>`
      }</tbody></table></div>
      <div class="pagination"><span>Page ${page + 1} of ${pages}</span><div>${paginationButton("left", `data-target-page="-1" ${page === 0 ? "disabled" : ""}`)}${paginationButton("right", `data-target-page="1" ${page + 1 === pages ? "disabled" : ""}`)}</div></div>`;
  };
  search.addEventListener("input", () => {
    page = 0;
    render();
  });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Enter") event.preventDefault();
  });
  results.addEventListener("click", (event) => {
    const control = event.target.closest("[data-target-page]");
    if (control) {
      page += Number(control.dataset.targetPage);
      render();
    }
  });
  render();
}
