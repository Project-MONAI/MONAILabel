import { escapeHTML as esc, button } from "./ui.js";

export function evaluationVersionOptions(state, selected = "") {
  return state.evaluationSets
    .filter(
      (s) =>
        !s.archived &&
        state.evaluationVersions.some((v) => v.evaluation_set_id === s.id),
    )
    .map(
      (s) =>
        `<optgroup label="${esc(s.name)}">${state.evaluationVersions
          .filter((v) => v.evaluation_set_id === s.id)
          .sort((a, b) => b.number - a.number)
          .map(
            (v) =>
              `<option value="${v.id}" ${selected === v.id ? "selected" : ""}>${esc(s.name)} · v${v.number} · ${v.samples.length} cases</option>`,
          )
          .join("")}</optgroup>`,
    )
    .join("");
}
export function evaluationVersionName(state, id) {
  const version = state.evaluationVersions.find((v) => v.id === id);
  const set = state.evaluationSets.find(
    (s) => s.id === version?.evaluation_set_id,
  );
  return version
    ? `${set?.name || "Evaluation set"} · v${version.number}`
    : "Dataset snapshot";
}

export async function evaluationSetAction(name, id, ui) {
  const { state, api, modal, field, refresh, render, notice, latestDecision } =
    ui;
  const prefix = `/projects/${state.project.id}/evaluation-sets`;
  const record = state.evaluationSets.find((s) => s.id === id);
  const manage = state.roles.includes("manager");
  const show = () => {
    const all = state.evaluationSets.filter(
      (s) => Boolean(s.archived) === Boolean(state.showArchivedSets),
    );
    const query = (state.evaluationSetSearch || "").toLowerCase();
    const matches = all.filter((s) => s.name.toLowerCase().includes(query));
    const pageCount = Math.max(1, Math.ceil(matches.length / 10));
    state.evaluationSetPage = Math.min(state.evaluationSetPage || 1, pageCount);
    const list = matches.slice(
      (state.evaluationSetPage - 1) * 10,
      state.evaluationSetPage * 10,
    );
    modal(
      "Evaluation sets",
      `<p>Reuse these sets across models. Saved versions keep comparisons on the same cases and reference labels.</p><div class="toolbar">${manage ? button("Create evaluation set", "evaluation-set-create", "", "primary") + button("Import files", "evaluation-import") : ""}${button(state.showArchivedSets ? "Show active" : "Show archived", "evaluation-set-toggle")}</div><label>Find a set<input type="search" id="evaluation-set-search" value="${esc(state.evaluationSetSearch || "")}"></label>` +
        (list
          .map((s) => {
            const members = state.assets.filter((a) =>
              s.member_groups.includes(a.group_id),
            );
            const pending = members.filter(
              (a) => latestDecision(a)?.verdict !== "accepted",
            ).length;
            const versions = state.evaluationVersions.filter(
              (v) => v.evaluation_set_id === s.id,
            );
            return `<section class="evaluation-set-card"><h3>${esc(s.name)}</h3><p>${members.length} evaluation-only cases${s.archived ? " · Archived" : ""}</p><p class="muted">${pending ? `${pending} awaiting accepted reference labels.` : members.length ? "All cases have accepted references." : "No cases reserved yet."} ${versions.length} saved version${versions.length === 1 ? "" : "s"}.</p><div class="toolbar">${button("View cases", "evaluation-set-cases", s.id)}${manage ? button("Settings", "evaluation-set-edit", s.id) + (!s.archived ? button("Save evaluation references", "evaluation-set-publish", s.id) + (state.selectedFiles.size ? button("Add selected samples", "evaluation-set-extend", s.id) : "") : "") + button(s.archived ? "Restore" : "Archive", "evaluation-set-archive", s.id) + (!versions.length ? button("Delete", "evaluation-set-delete", s.id) : "") : ""}</div>${
              versions.length
                ? `<details><summary>Version history</summary>${versions
                    .slice()
                    .reverse()
                    .map(
                      (v) =>
                        `<p>v${v.number} · ${v.samples.length} cases · ${esc(new Date(v.created_at).toLocaleDateString())}</p>`,
                    )
                    .join("")}</details>`
                : ""
            }</section>`;
          })
          .join("") || '<p class="muted">No matching evaluation sets.</p>') +
        `<div class="pagination"><span>${matches.length} sets · Page ${state.evaluationSetPage} of ${pageCount}</span><div>${state.evaluationSetPage > 1 ? button("Previous", "evaluation-set-page", String(state.evaluationSetPage - 1)) : ""}${state.evaluationSetPage < pageCount ? button("Next", "evaluation-set-page", String(state.evaluationSetPage + 1)) : ""}</div></div>`,
      async () => true,
      "Close",
    );
    document.querySelector("#evaluation-set-search").oninput = (event) => {
      const position = event.target.selectionStart;
      state.evaluationSetSearch = event.target.value;
      state.evaluationSetPage = 1;
      show();
      const input = document.querySelector("#evaluation-set-search");
      input.focus();
      input.setSelectionRange(position, position);
    };
  };
  if (name === "evaluation-sets") return show();
  if (name === "evaluation-set-toggle") {
    state.showArchivedSets = !state.showArchivedSets;
    state.evaluationSetPage = 1;
    return show();
  }
  if (name === "evaluation-set-page") {
    state.evaluationSetPage = Number(id);
    return show();
  }
  if (name === "evaluation-set-cases") {
    document.querySelector("#dialog").close();
    state.page = "datasets";
    state.datasetFilter = `set:${id}`;
    state.pages.datasets = 1;
    return render();
  }
  if (!manage) throw new Error("Project manager access is required.");
  if (
    ["evaluation-set-create", "evaluation-set-extend"].includes(name) &&
    (state.videos || []).some((video) => state.selectedFiles.has(video.id))
  )
    throw new Error(
      "Fixed evaluation references support images. Select image samples; video tracking evaluation is not available.",
    );
  if (name === "evaluation-set-create" || name === "evaluation-set-edit") {
    const editing = Boolean(record);
    const ids = [...state.selectedFiles];
    return modal(
      editing ? `Edit ${record.name}` : "Create evaluation set",
      field(
        "Set name",
        "name",
        "text",
        `required maxlength="120" value="${esc(record?.name || "Evaluation set " + (state.evaluationSets.length + 1))}"`,
      ) +
        `<p class="muted">${editing ? "Existing cases stay reserved." : ids.length ? `Reserves all ${ids.length} selected samples.` : "Creates an empty set. Import images + labels into it afterward."} Evaluation-only cases are excluded from every model's training. Model-specific percentages are set when starting training.</p>`,
      async (form) => {
        const body = {
          name: form.get("name").trim(),
          ...(editing ? {} : { percentage: 100 }),
          auto_update: false,
        };
        const saved = await api(
          editing ? `${prefix}/${id}` : prefix,
          editing ? "PATCH" : "POST",
          editing
            ? { ...body, base_version: record.version }
            : { ...body, asset_ids: ids },
        );
        if (editing) notice("Evaluation set updated.");
        else {
          const members = state.assets.filter((a) =>
            saved.member_groups.includes(a.group_id),
          );
          const pending = members.filter(
            (a) => latestDecision(a)?.verdict !== "accepted",
          ).length;
          notice(
            `${saved.name} created. ${members.length} case${members.length === 1 ? " is" : "s are"} reserved for evaluation. ` +
              (!members.length
                ? "Add samples to prepare this set."
                : pending
                  ? `${pending} still ${pending === 1 ? "needs" : "need"} accepted annotations in Reviews.`
                  : "Ready for model comparison."),
          );
        }
      },
      editing ? "Save settings" : "Create evaluation set",
    );
  }
  if (!record) throw new Error("Refresh and choose an evaluation set.");
  if (name === "evaluation-set-publish") {
    const labels = state.project.labels.filter((l) => l.id);
    return modal(
      `Save ${record.name} evaluation references`,
      '<p>Freezes the current cases and their accepted reference labels. Unchanged references reuse the existing version.</p><fieldset><legend>Structures to evaluate</legend><div class="checkboxes">' +
        labels
          .map(
            (l) =>
              `<label><input type="checkbox" name="labels" value="${l.id}" checked>${esc(l.name)}</label>`,
          )
          .join("") +
        "</div></fieldset>",
      async (form) => {
        const version = await api(`${prefix}/${id}/versions`, "POST", {
          base_version: record.version,
          label_ids: [0, ...form.getAll("labels").map(Number)],
        });
        state.context.evaluation_version_id = version.id;
        notice(`${record.name} · v${version.number} is ready for comparison.`);
      },
      "Save evaluation references",
    );
  }
  if (name === "evaluation-set-delete")
    return modal(
      `Delete ${record.name}?`,
      "<p>Deletes this unused set name. Its evaluation cases remain reserved; no images or annotations are deleted. Sets with saved versions can be archived instead.</p>",
      async () => {
        await api(`${prefix}/${id}`, "DELETE", {
          base_version: record.version,
        });
        notice("Unused evaluation set deleted.");
      },
      "Delete set",
    );
  if (name === "evaluation-set-extend")
    await api(`${prefix}/${id}/extend`, "POST", {
      base_version: record.version,
      asset_ids: [...state.selectedFiles],
      include_all: true,
    });
  if (name === "evaluation-set-archive")
    await api(`${prefix}/${id}`, "PATCH", {
      base_version: record.version,
      archived: !record.archived,
    });
  await refresh();
  show();
}
