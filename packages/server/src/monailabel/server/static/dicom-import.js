import { paginationButton, actionLabel } from "./icons.js";
import { escapeHTML as esc } from "./ui.js";

const field = (label, name, type = "text", attributes = "") =>
  `<label>${label}<input name="${name}" type="${type}" ${attributes}></label>`;
const dateText = (value) =>
  /^\d{8}$/.test(value)
    ? `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6)}`
    : value || "—";
const key = (row) => `${row.study_uid}/${row.series_uid}`;
const eligible = (row) => !row.imported_asset_id && !row.unsupported_reason;

export async function openDicomImport({ api, projectId, onImport }) {
  const prefix = `/projects/${projectId}/dicom-connections`;
  const settings = await api(prefix);
  const dialog = document.querySelector("#dialog");
  const form = document.querySelector("#action-form");
  const close = document.querySelector("#close-dialog");
  let busy = false;
  const run = async (operation) => {
    if (busy) return;
    busy = true;
    const error = form.querySelector(".form-error");
    error.textContent = "";
    const controls = [...form.querySelectorAll("input,select,button")].map(
      (node) => [node, node.disabled],
    );
    controls.forEach(([node]) => (node.disabled = true));
    close.disabled = true;
    dialog.dataset.busy = "true";
    try {
      await operation();
    } catch (reason) {
      form.querySelector(".form-error").textContent = reason.message;
    } finally {
      controls.forEach(([node, disabled]) => (node.disabled = disabled));
      busy = false;
      close.disabled = false;
      delete dialog.dataset.busy;
      form.dispatchEvent(new Event("dicom-ready"));
    }
  };
  const show = (title, html, wide = false) => {
    document.querySelector("#dialog-title").textContent = title;
    dialog.classList.toggle("dicom-browser", wide);
    form.innerHTML = html;
    form.onsubmit = (event) => event.preventDefault();
    if (!dialog.open) dialog.showModal();
  };
  dialog.addEventListener(
    "close",
    () => dialog.classList.remove("dicom-browser"),
    { once: true },
  );

  function connect(selectedId = settings.connections.at(-1)?.id || "") {
    const saved = settings.connections.find((item) => item.id === selectedId);
    show(
      "Connect to DICOM server",
      `
      <p class="muted">Enter your DICOM server’s web address. Ask its administrator for the DICOMweb URL if needed.</p>
      ${settings.connections.length ? `<label>Saved server<select name="saved"><option value="">Add a server</option>${settings.connections.map((c) => `<option value="${esc(c.id)}" ${c.id === selectedId ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select></label>` : ""}
      ${
        saved
          ? `<div class="dicom-source"><strong>${esc(saved.name)}</strong><span>${esc(saved.url)}</span><small>${saved.authentication === "none" ? "No sign-in required" : "Saved sign-in details"}</small></div>`
          : `
        ${field("Server URL (DICOMweb)", "url", "url", `required value="${esc(settings.suggested_url)}" placeholder="https://server.example/dicom-web"`)}
        <label>Sign-in method<select name="authentication"><option value="none">No sign-in required</option><option value="basic">Username and password</option><option value="bearer">Access token</option></select></label>
        <div id="dicom-auth"></div>
        <details><summary>Server name (optional)</summary>${field("Server name", "name", "text", 'maxlength="120" placeholder="e.g. Research archive"')}</details>`
      }
      <p class="form-error" role="alert"></p><button class="primary" type="submit">${actionLabel("Connect", "server")}</button>`,
    );
    form
      .querySelector('[name="saved"]')
      ?.addEventListener("change", (event) => connect(event.target.value));
    form
      .querySelector('[name="authentication"]')
      ?.addEventListener("change", (event) => {
        form.querySelector("#dicom-auth").innerHTML =
          event.target.value === "basic"
            ? field(
                "Username",
                "username",
                "text",
                'required autocomplete="username"',
              ) +
              field(
                "Password",
                "password",
                "password",
                'required autocomplete="current-password"',
              )
            : event.target.value === "bearer"
              ? field(
                  "Access token",
                  "token",
                  "password",
                  'required autocomplete="off"',
                )
              : "";
      });
    form.onsubmit = (event) => {
      event.preventDefault();
      const data = new FormData(form);
      run(async () => {
        const connection = saved
          ? await api(`${prefix}/${saved.id}/connect`, "POST")
          : await api(prefix, "POST", {
              name:
                data.get("name").trim() || new URL(data.get("url")).hostname,
              url: data.get("url").trim(),
              authentication: data.get("authentication"),
              username: data.get("username") || "",
              password: data.get("password") || "",
              token: data.get("token") || "",
            });
        if (!saved) settings.connections.push(connection);
        browse(connection);
      });
    };
  }

  function browse(connection) {
    let rows = [],
      selected = new Set(),
      page = 0,
      searched = false,
      truncated = false;
    const pageSize = window.matchMedia("(max-width: 700px)").matches ? 3 : 8;
    show(
      "Import DICOM series",
      `
      <div class="dicom-source-row"><div class="dicom-source"><strong>${esc(connection.name)} <span class="badge">Connected</span></strong><span>${esc(connection.url)}</span></div><button type="button" id="dicom-change" class="subtle">Change server</button></div>
      <p class="muted">Imported images are saved in this workspace and can open in Slicer or OHIF.</p>
      <div id="dicom-filters">
        <div class="dicom-filter-grid">
          <label>Image type (modality)<select name="modality"><option value="">All image types</option>${Object.entries(
            {
              CT: "CT",
              MR: "MRI",
              PT: "PET",
              US: "Ultrasound",
              CR: "Computed radiography",
              DX: "Digital X-ray",
              MG: "Mammography",
              NM: "Nuclear medicine",
              OT: "Other",
              SEG: "Segmentation",
              SR: "Structured report",
            },
          )
            .map(([value, name]) => `<option value="${value}">${name}</option>`)
            .join("")}</select></label>
          ${field("Study date from", "date_from", "date")}${field("Study date to", "date_to", "date")}
          <label>Import status<select name="import_status"><option value="new">Not imported</option><option value="all">All series</option><option value="imported">Already imported</option></select></label>
        </div>
        <details><summary>More filters</summary><div class="dicom-filter-grid">
          ${field("Patient ID", "patient_id")}${field("Patient name", "patient_name")}${field("Study description", "study_description")}${field("Series description", "series_description")}${field("Accession number", "accession_number")}
        </div><details><summary>Find by DICOM identifier</summary><div class="dicom-filter-grid">${field("Study unique ID (UID)", "study_uid")}${field("Series unique ID (UID)", "series_uid")}</div></details></details>
      </div>
      <div class="toolbar dicom-search-actions"><button type="button" id="dicom-search">${actionLabel("Search", "search")}</button><button type="button" id="dicom-reset" class="subtle">Reset filters</button></div>
      <div id="dicom-results" aria-live="polite"><p class="muted">Search to browse available series.</p></div>
      <p class="form-error" role="alert"></p>
      <div class="dicom-import-actions"><button type="button" id="dicom-all" disabled>${actionLabel("Import all matches", "upload")}</button><button class="primary" type="submit" disabled>${actionLabel("Import selected", "upload")}</button></div>`,
      true,
    );
    const submit = form.querySelector('[type="submit"]');
    const all = form.querySelector("#dicom-all");
    const results = form.querySelector("#dicom-results");
    const updateActions = () => {
      submit.disabled = busy || !selected.size || !searched;
      all.disabled = busy || truncated || !searched || !rows.some(eligible);
      submit.innerHTML = actionLabel(
        selected.size
          ? `Import selected (${selected.size})`
          : "Import selected",
        "upload",
      );
    };
    form.addEventListener("dicom-ready", updateActions);
    dialog.addEventListener(
      "close",
      () => form.removeEventListener("dicom-ready", updateActions),
      { once: true },
    );
    const render = () => {
      const pages = Math.max(1, Math.ceil(rows.length / pageSize));
      page = Math.min(page, pages - 1);
      results.innerHTML = `<div class="selection-bar"><span>${rows.length} matching series · ${rows.filter(eligible).length} available to import</span></div>
        ${truncated ? '<p class="form-error">More than 1,000 series match. Refine the filters before importing all matches.</p>' : ""}
        ${
          rows.length
            ? `<div class="table-wrap"><table class="compact-table dicom-results"><thead><tr><th><input type="checkbox" id="dicom-select-page" aria-label="Select available series on this page"></th><th>Patient / study</th><th>Series</th><th>Modality</th><th>Images</th><th>Status</th></tr></thead><tbody>${rows
                .slice(page * pageSize, (page + 1) * pageSize)
                .map(
                  (row) => `<tr>
          <td class="check-cell"><input type="checkbox" data-dicom-series="${esc(key(row))}" aria-label="Select ${esc(row.series_description || row.series_uid)}" ${eligible(row) ? "" : "disabled"} ${selected.has(key(row)) ? "checked" : ""}></td>
          <td class="dicom-patient"><strong>${esc(row.patient_name.replaceAll("^", " ") || row.patient_id || "Unknown patient")}</strong><small>${esc(row.patient_id)} · ${esc(dateText(row.study_date))}</small><small>${esc(row.study_description)}</small></td>
          <td class="dicom-series"><strong>${esc(row.series_description || "Unnamed series")}</strong><small title="${esc(row.series_uid)}">${esc(row.series_uid)}</small></td>
          <td data-label="Modality">${esc(row.modality || "—")}</td><td data-label="Images">${row.instance_count ?? "—"}</td>
          <td class="dicom-status">${row.imported_asset_id ? "Imported" : row.unsupported_reason ? `<span title="${esc(row.unsupported_reason)}">Unsupported</span><small>${esc(row.unsupported_reason)}</small>` : "Ready"}</td></tr>`,
                )
                .join("")}</tbody></table></div>
          <div class="pagination"><span>Page ${page + 1} of ${pages}</span><div>${paginationButton("left", `data-dicom-page="-1" ${page === 0 ? "disabled" : ""}`)}${paginationButton("right", `data-dicom-page="1" ${page + 1 === pages ? "disabled" : ""}`)}</div></div>`
            : '<p class="empty">No series match these filters.</p>'
        }`;
      const available = rows
        .slice(page * pageSize, (page + 1) * pageSize)
        .filter(eligible);
      const selectPage = results.querySelector("#dicom-select-page");
      if (selectPage) {
        selectPage.disabled = !available.length;
        selectPage.checked =
          available.length > 0 &&
          available.every((row) => selected.has(key(row)));
        selectPage.indeterminate =
          available.some((row) => selected.has(key(row))) &&
          !selectPage.checked;
      }
      updateActions();
    };
    const search = () => {
      const data = new FormData(form);
      const filters = Object.fromEntries(
        [...data].filter(([name, value]) => value && !name.startsWith("dicom")),
      );
      searched = false;
      selected.clear();
      rows = [];
      updateActions();
      run(async () => {
        results.innerHTML =
          '<p class="muted" role="status">Searching the DICOM server…</p>';
        let response;
        try {
          response = await api(
            `${prefix}/${connection.id}/search`,
            "POST",
            filters,
          );
        } catch (error) {
          results.innerHTML =
            '<p class="muted">Search failed. Check the filters and connection, then try again.</p>';
          throw error;
        }
        rows = response.items;
        truncated = response.truncated;
        selected.clear();
        page = 0;
        searched = true;
        render();
      });
    };
    const importRows = (items) =>
      run(async () => {
        const job = await api(`${prefix}/${connection.id}/imports`, "POST", {
          series: items.map(({ study_uid, series_uid }) => ({
            study_uid,
            series_uid,
          })),
        });
        dialog.close();
        await onImport(job);
      });
    form.querySelector("#dicom-search").onclick = search;
    form.querySelector("#dicom-change").onclick = () => {
      form.removeEventListener("dicom-ready", updateActions);
      connect(connection.id);
    };
    form.querySelector("#dicom-reset").onclick = () => {
      form
        .querySelectorAll("#dicom-filters input")
        .forEach((input) => (input.value = ""));
      form.querySelector('[name="modality"]').value = "";
      form.querySelector('[name="import_status"]').value = "new";
      search();
    };
    form.querySelector("#dicom-filters").addEventListener("input", () => {
      searched = false;
      selected.clear();
      rows = [];
      results.innerHTML =
        '<p class="muted">Filters changed. Search to update the results.</p>';
      updateActions();
    });
    form
      .querySelector("#dicom-filters")
      .addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          search();
        }
      });
    results.addEventListener("change", (event) => {
      const input = event.target;
      const keys =
        input.id === "dicom-select-page"
          ? rows
              .slice(page * pageSize, (page + 1) * pageSize)
              .filter(eligible)
              .map(key)
          : [input.dataset.dicomSeries];
      keys
        .filter(Boolean)
        .forEach((id) =>
          input.checked ? selected.add(id) : selected.delete(id),
        );
      render();
    });
    results.addEventListener("click", (event) => {
      const control = event.target.closest("[data-dicom-page]");
      if (control) {
        page += Number(control.dataset.dicomPage);
        render();
      }
    });
    all.onclick = () => {
      if (searched && !truncated && rows.some(eligible))
        importRows(rows.filter(eligible));
    };
    form.onsubmit = (event) => {
      event.preventDefault();
      if (searched && selected.size)
        importRows(
          rows.filter((row) => selected.has(key(row)) && eligible(row)),
        );
    };
  }
  connect();
}
