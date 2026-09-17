import { paginationButton } from "./icons.js";
import { reviewDecisionControl } from "./review-controls.js";
// Compact project lists. Page/search/filter state stays with the application controller.
import {
  escapeHTML as esc,
  badge,
  button,
  jobName,
  jobFinished,
} from "./ui.js";

const PAGE_SIZES = { datasets: 10, review: 10, activity: 10 };
const matches = (value, query) =>
  value.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());
export function pageSlice(items, state, key) {
  const pageSize = PAGE_SIZES[key];
  const count = Math.max(1, Math.ceil(items.length / pageSize));
  state.pages[key] = Math.max(1, Math.min(state.pages[key], count));
  return items.slice(
    (state.pages[key] - 1) * pageSize,
    state.pages[key] * pageSize,
  );
}
function pagination(items, state, key) {
  const pageSize = PAGE_SIZES[key];
  const page = state.pages[key];
  const pages = Math.max(1, Math.ceil(items.length / pageSize));
  return `<div class="pagination"><span>${items.length ? `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, items.length)} of ${items.length}` : "0 results"}</span><div>${paginationButton("left", `data-action="page" data-id="${page - 1}" ${page === 1 ? "disabled" : ""}`)}<span>Page ${page} of ${pages}</span>${paginationButton("right", `data-action="page" data-id="${page + 1}" ${page === pages ? "disabled" : ""}`)}</div></div>`;
}
function filters(state, key, options, value, hint) {
  return `<div class="list-filters"><label class="search-field"><span class="sr-only">Search ${key}</span><input type="search" data-search="${key}" placeholder="${hint}" value="${esc(state.searches[key])}"></label><label><span class="sr-only">Filter ${key}</span><select id="${key === "datasets" ? "dataset" : key}-filter" aria-label="Filter ${key}">${options.map(([id, label]) => `<option value="${esc(id)}" ${value === id ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label></div>`;
}
const table = (headings, rows) =>
  `<div class="table-wrap"><table class="compact-table"><thead><tr>${headings.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
const empty = (text, reset = false) =>
  `<div class="empty"><p>${text}</p>${reset ? button("Reset filters", "reset-filters") : ""}</div>`;
const viewerButtons = (state, a) =>
  `<div class="row-actions">${button(a.kind === "volume3d" ? "Slicer" : "QuPath", "viewer", a.id)}${a.kind === "volume3d" ? button("OHIF", "ohif", a.id) : ""}</div>`;
const sampleActions = (asset) =>
  `<button class="sample-actions" data-action="sample-details" data-id="${esc(asset.id)}" type="button" title="Sample actions" aria-haspopup="dialog" aria-label="Actions for ${esc(asset.name)}"><span aria-hidden="true">⋯</span></button>`;
export function visibleAssets(state, statusOf) {
  return state.assets.filter(
    (a) =>
      matches(`${a.name} ${a.group_id}`, state.searches.datasets) &&
      (state.datasetFilter === "all" ||
        a.split === state.datasetFilter ||
        (state.datasetFilter === "shared" && a.split !== "validation") ||
        statusOf(a) === state.datasetFilter ||
        (state.datasetFilter.startsWith("set:") &&
          state.evaluationSets
            .find((s) => s.id === state.datasetFilter.slice(4))
            ?.member_groups.includes(a.group_id))),
  );
}
export function datasets(state, manage, statusOf) {
  const items = visibleAssets(state, statusOf);
  const rows = pageSlice(items, state, "datasets");
  const options = [
    ["all", "All samples"],
    ["shared", "Annotation & training"],
    ["validation", "Evaluation only"],
    ["unannotated", "Not submitted"],
    ["pending", "Pending review"],
    ["accepted", "Accepted"],
    ["changes_requested", "Needs changes"],
  ];
  options.push(
    ...state.evaluationSets
      .filter((s) => !s.archived || state.datasetFilter === `set:${s.id}`)
      .map((s) => [`set:${s.id}`, s.name]),
  );
  const selected = state.selectedFiles.size;
  return (
    `<div class="section-heading"><p class="muted">Annotate images, then use reviewed labels to train or fine-tune models. Each model manages its own validation split; evaluation-only data stays separate.</p><div class="toolbar">${manage ? button("Sample datasets", "dataset-template") + button("Import from DICOM server", "dicom") + button("Import files", "dataset", "", "primary") : ""}</div></div>` +
    filters(
      state,
      "datasets",
      options,
      state.datasetFilter,
      "Search samples, patients or slides",
    ) +
    (manage && state.assets.length
      ? `<div class="selection-bar"><span>${selected ? `${selected} selected` : `${items.length} ${items.length === 1 ? "sample" : "samples"}`}</span>${selected ? button("Clear selection", "clear-selection") : ""}${items.length > PAGE_SIZES.datasets ? button(`Select all ${items.length} matching`, "select-filtered") : ""}${selected ? button("Delete selected files", "delete-files", "", "danger") : ""}</div>`
      : "") +
    (rows.length
      ? table(
          [
            ...(manage
              ? [
                  '<input type="checkbox" id="select-all-files" aria-label="Select files on this page">',
                ]
              : []),
            "Sample",
            "Status",
            "Actions",
          ],
          rows.map(
            (a) =>
              `<tr class="${a.id === state.context.asset_id ? "row-selected" : ""}">${manage ? `<td class="check-cell"><input type="checkbox" data-file-selection="${a.id}" aria-label="Select ${esc(a.name)}" ${state.selectedFiles.has(a.id) ? "checked" : ""}></td>` : ""}<td class="sample-cell">${button(esc(a.name), "select", a.id, "link-button")}<small>${esc(a.spatial_shape.join(" × "))} · Revision ${a.revision}</small></td><td>${badge(a.split === "validation" ? "Evaluation only" : "Annotation & training")} ${badge(statusOf(a))}</td><td><div class="row-actions">${viewerButtons(state, a)}${sampleActions(a)}</div></td></tr>`,
          ),
        )
      : empty(
          state.assets.length
            ? "No samples match these filters."
            : "Import files to add your first sample.",
          !!state.assets.length,
        )) +
    pagination(items, state, "datasets")
  );
}
export function visibleReviews(state, statusOf) {
  return state.assets.filter(
    (a) =>
      a.annotation_id &&
      (state.reviewFilter === "all" || statusOf(a) === state.reviewFilter) &&
      matches(a.name, state.searches.review),
  );
}
export function reviewQueue(state, statusOf, latestDecision) {
  const canReview = state.roles.includes("reviewer");
  const selected = state.selectedReviews || new Set();
  const pending = state.assets.filter(
    (a) => a.annotation_id && statusOf(a) === "pending",
  );
  const submitted = state.assets.filter((a) => a.annotation_id);
  const items = visibleReviews(state, statusOf);
  const rows = pageSlice(items, state, "review");
  const options = [
    ["pending", "Pending review"],
    ["accepted", "Accepted"],
    ["changes_requested", "Needs changes"],
    ["all", "All reviews"],
  ].map(([id, label]) => [
    id,
    `${label} (${submitted.filter((a) => id === "all" || statusOf(a) === id).length})`,
  ]);
  return (
    `<p class="muted">Accept saved annotations here, or open a viewer to inspect and correct them.</p>` +
    (canReview && (items.length || selected.size)
      ? `<div class="selection-bar"><span>${selected.size} selected · ${pending.length} pending</span>${selected.size ? reviewDecisionControl() + button("Clear selection", "clear-review-selection") : ""}${items.length > PAGE_SIZES.review ? button(`Select all ${items.length} matching`, "select-filtered-reviews") : ""}</div>`
      : "") +
    filters(
      state,
      "review",
      options,
      state.reviewFilter,
      "Search review samples",
    ) +
    (rows.length
      ? table(
          [
            ...(canReview
              ? [
                  '<input type="checkbox" id="select-page-reviews" aria-label="Select reviews on this page">',
                ]
              : []),
            "Sample",
            "Review",
            "Actions",
          ],
          rows.map(
            (a) =>
              `<tr>${canReview ? `<td class="check-cell">${`<input type="checkbox" data-review-selection="${a.annotation_id}" aria-label="Select ${esc(a.name)}" ${selected.has(a.annotation_id) ? "checked" : ""}>`}</td>` : ""}<td class="sample-cell">${esc(a.name)}<small>Revision ${a.revision}</small></td><td>${badge(statusOf(a))}${latestDecision(a)?.comment ? `<p class="cell-note" title="${esc(latestDecision(a).comment)}">${esc(latestDecision(a).comment)}</p>` : ""}</td><td><div class="row-actions">${viewerButtons(state, a)}</div></td></tr>`,
          ),
        )
      : empty(
          submitted.length
            ? "No annotations match this review filter."
            : "Submitted annotations will appear here for review.",
          !!submitted.length,
        )) +
    pagination(items, state, "review")
  );
}
export function activity(state, canCancel) {
  const items = [...state.jobs]
    .reverse()
    .filter(
      (j) =>
        (state.activityFilter === "all" ||
          (state.activityFilter === "active"
            ? !jobFinished(j)
            : state.activityFilter === "errors"
              ? ["failed", "interrupted"].includes(j.status)
              : j.status === state.activityFilter)) &&
        matches(
          `${jobName(j.kind)} ${j.error || ""} ${j.progress_message || ""}`,
          state.searches.activity,
        ),
    );
  const rows = pageSlice(items, state, "activity");
  const active = state.jobs.filter((j) => !jobFinished(j)).length;
  return (
    `<div class="section-heading"><p class="muted">${active} active · ${state.jobs.length} total · Updates automatically</p></div>` +
    filters(
      state,
      "activity",
      [
        ["all", "All activity"],
        ["active", "In progress"],
        ["succeeded", "Completed"],
        ["errors", "Failed or interrupted"],
        ["cancelled", "Cancelled"],
      ],
      state.activityFilter,
      "Search activity",
    ) +
    (rows.length
      ? table(
          ["Task", "Status", "Started", "Actions"],
          rows.map(
            (j) =>
              `<tr><td class="task-cell"><strong>${esc(jobName(j.kind))}</strong><p class="cell-note" title="${esc(j.error || j.progress_message || "")}">${esc(j.error || j.progress_message || "")}</p></td><td>${badge(j.status)}${!jobFinished(j) ? `<progress value="${j.progress}" max="1" aria-label="${esc(jobName(j.kind))} progress"></progress><small>${Math.round(j.progress * 100)}%</small>` : ""}</td><td class="date-cell">${esc(new Date(j.created_at).toLocaleDateString())}<small>${esc(new Date(j.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }))}</small></td><td><div class="row-actions">${button(["train", "evaluate", "training_report", "batch_annotate"].includes(j.kind) ? (jobFinished(j) ? "Results" : "View logs") : "Details", "job-details", j.id)}${canCancel && !jobFinished(j) ? button("Cancel", "cancel", j.id) : ""}</div></td></tr>`,
          ),
        )
      : empty(
          state.jobs.length
            ? "No activity matches these filters."
            : "Annotation, training and viewer tasks will appear here.",
          !!state.jobs.length,
        )) +
    pagination(items, state, "activity")
  );
}
export function team(state, manage) {
  return (
    `<div class="section-heading"><p class="muted">Your access: ${esc(state.roles.map((r) => r[0].toUpperCase() + r.slice(1)).join(", "))}</p><div class="toolbar">${manage ? button("Assign roles", "member", "", "primary") : ""}${state.user.is_admin ? button("Create user", "user") : ""}</div></div>` +
    (manage
      ? state.members.length
        ? table(
            ["Member", "Roles", ""],
            state.members.map(
              (m) =>
                `<tr><td>${esc(m.username)}${m.active === false ? ` ${badge("Inactive")}` : ""}</td><td>${m.roles.map((r) => badge(r[0].toUpperCase() + r.slice(1))).join(" ")}</td><td>${button("Edit roles", "member", m.user_id)}</td></tr>`,
            ),
          )
        : empty(
            "No project members assigned. Administrators already have access.",
          )
      : "") +
    `<details class="card"><summary>About project roles</summary><dl class="details-list"><dt>Manager</dt><dd>Manage data, models, team and training; annotate and review.</dd><dt>Annotator</dt><dd>Generate, edit and submit annotations.</dd><dt>Reviewer</dt><dd>Review in the viewer, accept corrections or request changes.</dd></dl><p class="muted">Administrators have access to every project. Managers may also review their own work.</p></details>`
  );
}
