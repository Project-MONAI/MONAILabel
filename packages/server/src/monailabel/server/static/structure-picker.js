import { paginationButton } from "./icons.js";
import { escapeHTML as esc } from "./ui.js";

// Selection is independent of the visible search/page so choices never disappear.
export function structurePicker(container, recipe, limit = 31) {
  const targets = [...(recipe?.supported_targets || [])].sort((a, b) =>
    a.localeCompare(b),
  );
  const ids = recipe?.target_class_ids || {};
  const selected = new Set();
  const pageSize = 6;
  let page = 0;
  container.classList.add("structure-picker");
  container.innerHTML = `<label>Find a structure<input type="search" placeholder="Search by name or class ID…" autocomplete="off"></label>
    <div class="structure-selection"></div><div class="structure-results"></div>`;
  const search = container.querySelector('input[type="search"]');
  const selection = container.querySelector(".structure-selection");
  const results = container.querySelector(".structure-results");
  function render() {
    const query = search.value.trim().toLowerCase();
    const matches = targets.filter((name) =>
      /^\d+$/.test(query)
        ? ids[name.toLowerCase()] === Number(query)
        : name.toLowerCase().includes(query),
    );
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.max(0, Math.min(page, pages - 1));
    selection.innerHTML = selected.size
      ? `<div class="structure-selection-heading"><span>${selected.size} selected${selected.size === limit ? ` · limit ${limit}` : ""}</span><button type="button" class="model-text-action" data-clear-structures>Clear selection</button></div><div class="structure-chips">${[...selected].map((name) => `<button type="button" data-remove-structure="${esc(name)}" aria-label="Remove ${esc(name)}">${esc(name[0].toUpperCase() + name.slice(1))}<span aria-hidden="true">×</span></button>`).join("")}</div>`
      : `<p class="muted">None selected · choose training organs later.</p>`;
    results.innerHTML = `<p class="muted" role="status">${matches.length} of ${targets.length} structures</p>
      <div class="structure-table"><table aria-label="VISTA3D training structures"><thead><tr><th scope="col">Structure</th><th scope="col" class="target-class-id">Class ID</th></tr></thead><tbody>${
        matches
          .slice(page * pageSize, (page + 1) * pageSize)
          .map(
            (name) =>
              `<tr><td><label><input type="checkbox" data-structure value="${esc(name)}" ${selected.has(name) ? "checked" : selected.size >= limit ? "disabled" : ""}>${esc(name[0].toUpperCase() + name.slice(1))}</label></td><td class="target-class-id">${ids[name.toLowerCase()] ?? "—"}</td></tr>`,
          )
          .join("") || '<tr><td colspan="2">No matching structures</td></tr>'
      }</tbody></table></div>
      <div class="pagination"><span>Page ${page + 1} of ${pages}</span><div>${paginationButton("left", `data-structure-page="-1" ${page === 0 ? "disabled" : ""}`)}${paginationButton("right", `data-structure-page="1" ${page + 1 === pages ? "disabled" : ""}`)}</div></div>`;
  }
  search.addEventListener("input", () => {
    page = 0;
    render();
  });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Enter") event.preventDefault();
  });
  container.addEventListener("change", (event) => {
    const input = event.target.closest("[data-structure]");
    if (!input) return;
    if (input.checked && selected.size < limit) selected.add(input.value);
    else selected.delete(input.value);
    const name = input.value;
    render();
    [...results.querySelectorAll("[data-structure]")]
      .find((element) => element.value === name)
      ?.focus({ preventScroll: true });
  });
  container.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.hasAttribute("data-clear-structures")) selected.clear();
    else if (button.hasAttribute("data-remove-structure"))
      selected.delete(button.dataset.removeStructure);
    else if (button.hasAttribute("data-structure-page"))
      page += Number(button.dataset.structurePage);
    else return;
    render();
  });
  render();
  return () => [...selected];
}
