// Hosted discovery uses server-held credentials and fixed provider endpoints.
export async function importHostedModel(ui, back, preset = null) {
  const {
    state,
    api,
    modal,
    field,
    selectField,
    escapeHTML: esc,
    message,
  } = ui;
  const prefix = `/projects/${state.project.id}`;
  const catalogPrefix = preset ? `${prefix}/models/${preset.id}` : prefix;
  let services = [];
  let catalog = [];
  let requestVersion = 0;
  let catalogConnection = null;

  modal(
    preset ? `Provider for ${preset.name}` : "Import a hosted model",
    back +
      `<div id="hosted-import">
      <div id="hosted-service"></div>
      <div id="hosted-key"></div>
      <p id="catalog-status" class="muted" role="status">Loading model services…</p>
      <div id="catalog-results" hidden>
        ${field("Search models", "model_search", "search", 'autocomplete="off" placeholder="Name or model ID"')}
        <label><span id="catalog-model-label">Model</span>
          <select name="model" size="7" aria-labelledby="catalog-model-label"></select>
        </label>
        <p id="catalog-selection" class="muted"></p>
        ${field("Model name", "name", "text", `required maxlength="120" ${preset ? "readonly" : ""}`, preset ? "The preset keeps its name." : "Use this name in annotation prompts. Added only to this project.")}
        <details><summary>Advanced options</summary>
          ${field("Response limit (tokens)", "max_output_tokens", "number", 'required value="4096" min="64" max="16384"', "Detailed outlines may need a higher limit; some models share this allowance with reasoning.")}
        </details>
      </div>
      ${preset ? "" : '<button type="button" class="subtle" data-action="model" data-id="custom">Connect another vision API</button>'}
    </div>`,
    async (f) => {
      if (preset && f.get("service") === "automatic") {
        const updated = await api(`${catalogPrefix}/provider`, "POST", {
          base_version: preset.version,
        });
        replaceSelection(updated);
        message(
          "Automatic provider selection restored. NVIDIA is preferred when available.",
        );
        return;
      }
      if (!catalogConnection || !catalog.some((m) => m.id === f.get("model")))
        throw new Error("Load the model list and select a model first.");
      const connection = {
        ...catalogConnection,
        model: f.get("model"),
        name: f.get("name").trim(),
        max_output_tokens: Number(f.get("max_output_tokens")),
      };
      if (preset) {
        const updated = await api(`${catalogPrefix}/provider`, "POST", {
          base_version: preset.version,
          connection,
        });
        replaceSelection(updated);
      } else await api(`${prefix}/model-import`, "POST", connection);
      state.page = "models";
      message(
        preset
          ? "Provider changed for this project."
          : "Model added. Select it in your viewer or name it in an annotation prompt.",
      );
    },
    preset ? "Change provider" : "Add model",
  );
  const form = document.querySelector("#action-form");
  const root = form.querySelector("#hosted-import");
  const status = root.querySelector("#catalog-status");
  const results = root.querySelector("#catalog-results");
  const submit = form.querySelector('button[type="submit"]');
  const active = () =>
    root.isConnected && document.querySelector("#dialog").open;
  const service = () =>
    services.find((s) => s.id === form.elements.service.value);
  const current = (version) => active() && version === requestVersion;
  submit.disabled = true;

  function replaceSelection(updated) {
    for (const key of ["model_id", "baseline_id"])
      if (state.context[key] === preset.id) state.context[key] = updated.id;
  }

  function invalidate() {
    requestVersion++;
    catalog = [];
    catalogConnection = null;
    results.hidden = true;
    submit.disabled = true;
    form.querySelector(".form-error").textContent = "";
    form.elements.model_search.value = "";
    form.elements.model.innerHTML = "";
    form.elements.name.value = preset ? preset.name : "";
    const refresh = root.querySelector("#load-models");
    if (refresh) refresh.disabled = false;
  }

  function selectionChanged(updateName = false) {
    const selected = catalog.find((m) => m.id === form.elements.model.value);
    submit.disabled = !selected || !!selected.existing_model_id;
    root.querySelector("#catalog-selection").textContent = selected
      ? selected.existing_model_id
        ? "Already added to this project."
        : selected.id
      : "Select a model to add to this project.";
    if (updateName && selected)
      form.elements.name.value = preset
        ? preset.name
        : selected.name.slice(0, 120);
  }

  function filterModels() {
    const selected = form.elements.model.value;
    const query = form.elements.model_search.value.trim().toLowerCase();
    const matches = catalog.filter((m) =>
      `${m.name} ${m.id}`.toLowerCase().includes(query),
    );
    form.elements.model.size = Math.max(2, Math.min(7, matches.length));
    form.elements.model.innerHTML = matches
      .map(
        (m) =>
          `<option value="${esc(m.id)}">${esc(modelLabel(m))}${m.existing_model_id ? " · Already added" : ""}</option>`,
      )
      .join("");
    form.elements.model.value = matches.some((m) => m.id === selected)
      ? selected
      : "";
    selectionChanged();
    if (!matches.length)
      root.querySelector("#catalog-selection").textContent =
        "No models match your search.";
  }

  function modelLabel(model) {
    if (form.elements.service.value !== "nvidia") return model.name;
    const parts = model.id.split("/");
    const name = model.name === model.id ? parts.at(-1) : model.name;
    const route = parts.slice(0, -1).join("/");
    return route ? `${name} · ${route}` : name;
  }

  function renderKey() {
    const selected = service();
    const source = form.elements.auth.value;
    root.querySelector("#key-details").innerHTML =
      source === "environment"
        ? `<p class="muted">${esc(selected.token_env)} ${selected.configured ? "is configured on the server." : "is not configured. Set it before starting the server, or choose another key source."}</p>`
        : source === "saved"
          ? selectField(
              "Saved API key",
              "credential",
              '<option value="">Choose a key</option>' +
                state.credentials
                  .map(
                    (c) =>
                      `<option value="${esc(c.id)}">${esc(c.name)}</option>`,
                  )
                  .join(""),
            )
          : field(
              "API key",
              "api_key",
              "password",
              'autocomplete="off"',
              "Saved encrypted on this server when you load models.",
            );
    const control = form.elements.credential || form.elements.api_key;
    if (control)
      control.addEventListener(source === "saved" ? "change" : "input", () => {
        invalidate();
        status.textContent = "Load models with this key.";
        if (source === "saved" && control.value) loadModels();
      });
  }

  async function loadModels() {
    invalidate();
    const version = requestVersion;
    const selected = service();
    const source = form.elements.auth.value;
    const refresh = root.querySelector("#load-models");
    refresh.disabled = true;
    status.textContent = "Loading compatible models…";
    try {
      const connection = { service: selected.id };
      if (source === "saved") {
        connection.credential_id = form.elements.credential.value;
        if (!connection.credential_id)
          throw new Error("Choose a saved API key first.");
      } else if (source === "new") {
        const key = form.elements.api_key.value.trim();
        if (!key) throw new Error("Enter an API key first.");
        const saved = await api(`${prefix}/credentials`, "POST", {
          name: `${selected.name} key`,
          api_key: key,
        });
        if (!current(version)) return;
        state.credentials.push(saved);
        form.elements.auth.value = "saved";
        renderKey();
        form.elements.credential.value = saved.id;
        connection.credential_id = saved.id;
      }
      const response = await api(
        `${catalogPrefix}/model-catalog`,
        "POST",
        connection,
      );
      if (!current(version)) return;
      catalogConnection = connection;
      catalog = response.models;
      status.textContent = catalog.length
        ? `${catalog.length} compatible model${catalog.length === 1 ? "" : "s"}.${response.excluded ? ` ${response.excluded} other models hidden.` : ""}`
        : "No compatible annotation models are available with this key.";
      results.hidden = !catalog.length;
      filterModels();
      if (preset && catalog.length === 1) {
        form.elements.model.value = catalog[0].id;
        selectionChanged(true);
      }
    } catch (error) {
      if (current(version)) status.textContent = error.message;
    } finally {
      if (current(version)) refresh.disabled = false;
    }
  }

  function serviceChanged() {
    invalidate();
    if (preset && !form.elements.service.value) {
      status.textContent =
        "Choose a provider, or Automatic to prefer NVIDIA. This changes only this project.";
      return;
    }
    if (preset && form.elements.service.value === "automatic") {
      root.querySelector("#hosted-key").innerHTML = "";
      status.textContent =
        "Use NVIDIA when available, otherwise the model’s direct provider. This uses keys configured on the server.";
      submit.disabled = false;
      return;
    }
    root.querySelector("#hosted-key").innerHTML =
      selectField(
        "API key source",
        "auth",
        '<option value="environment">Key configured on the server</option><option value="saved">Use a saved API key</option><option value="new">Save a new API key</option>',
      ) +
      '<div id="key-details"></div><button type="button" id="load-models">Load models</button>';
    const selected = service();
    form.elements.auth.value = selected.configured ? "environment" : "new";
    form.elements.auth.addEventListener("change", () => {
      invalidate();
      renderKey();
      status.textContent = "Choose a key to load compatible annotation models.";
      if (form.elements.auth.value === "environment" && selected.configured)
        loadModels();
    });
    root.querySelector("#load-models").addEventListener("click", loadModels);
    renderKey();
    status.textContent = "Choose a key to load compatible annotation models.";
    if (selected.configured) loadModels();
  }

  form.elements.model_search.addEventListener("input", filterModels);
  form.elements.model.addEventListener("change", () => selectionChanged(true));
  try {
    services = await api(`${catalogPrefix}/model-services`);
    if (!active()) return;
    root.querySelector("#hosted-service").innerHTML = selectField(
      "Model service",
      "service",
      (preset
        ? '<option value="" disabled>Choose a provider</option><option value="automatic">Automatic · NVIDIA first</option>'
        : "") +
        services
          .map(
            (s) =>
              `<option value="${esc(s.id)}">${esc(s.name)}${s.configured ? " · Key configured" : ""}</option>`,
          )
          .join(""),
    );
    form.elements.service.value = preset
      ? ""
      : (services.find((s) => s.configured) || services[0]).id;
    if (preset)
      form.elements.max_output_tokens.value =
        preset.config.max_output_tokens || 16384;
    form.elements.service.addEventListener("change", serviceChanged);
    serviceChanged();
  } catch (error) {
    if (active()) status.textContent = error.message;
  }
}
