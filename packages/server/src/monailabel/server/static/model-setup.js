// Guided model setup. Inference connections and trainable recipes have separate forms.
import { importHostedModel } from "./model-import.js";

export function setupModel(ui, path = "") {
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
  const labels = () =>
    `<fieldset><legend>Structures this model supports</legend><div class="checkboxes">${state.project.labels
      .filter((l) => l.id)
      .map(
        (l) =>
          `<label><input type="checkbox" name="labels" value="${l.id}" checked>${esc(l.name)}</label>`,
      )
      .join(
        "",
      )}</div><small>Background is included automatically.</small></fieldset>`;
  const labelIds = (f, promptable = false) => {
    const ids = f.getAll("labels").map(Number);
    if (!ids.length && !promptable)
      throw new Error("Select at least one structure.");
    return [0, ...ids];
  };
  const credentials = () =>
    selectField(
      "API key source",
      "auth",
      '<option value="environment">Key configured on the server</option><option value="saved">Use a saved API key</option><option value="new">Save a new API key</option><option value="none">No API key needed</option>',
    ) + '<div id="model-auth"></div>';
  function bindCredentials(defaultEnv) {
    const form = document.querySelector("#action-form");
    function change() {
      const mode = form.elements.auth.value;
      form.querySelector("#model-auth").innerHTML =
        mode === "environment"
          ? field(
              "Environment variable name",
              "token_env",
              "text",
              `required value="${esc(typeof defaultEnv === "function" ? defaultEnv() : defaultEnv)}"`,
              "Enter the variable name, not the secret. The server resolves it when the model runs.",
            )
          : mode === "saved"
            ? selectField(
                "Saved API key",
                "credential",
                state.credentials
                  .map((c) => `<option value="${c.id}">${esc(c.name)}</option>`)
                  .join(""),
              )
            : mode === "new"
              ? field(
                  "API key",
                  "api_key",
                  "password",
                  "required autocomplete='off'",
                  "Encrypted on this server. Never enter keys in chat.",
                )
              : "";
    }
    form.elements.auth.addEventListener("change", change);
    change();
  }
  async function addCredentials(f, config, name) {
    if (f.get("auth") === "environment")
      config.token_env = f.get("token_env").trim();
    if (f.get("auth") === "saved") {
      if (!f.get("credential"))
        throw new Error("Save a key first or choose another API key source.");
      config.credential_id = f.get("credential");
    }
    if (f.get("auth") === "new") {
      const saved = await api(`${prefix}/credentials`, "POST", {
        name: `${name} key`,
        api_key: f.get("api_key"),
      });
      config.credential_id = saved.id;
    }
  }
  async function register(f, provider, config, name) {
    const ids = labelIds(
      f,
      [
        "openai-polygons",
        "openai-chat-polygons",
        "anthropic-polygons",
      ].includes(provider),
    );
    await addCredentials(f, config, name);
    await api(`${prefix}/models`, "POST", {
      name,
      provider,
      config,
      label_ids: ids,
    });
    state.page = "models";
    message(
      "Model connected. Select it in the workspace or your viewer to annotate a sample.",
    );
  }
  if (!path) {
    modal(
      "Add a model",
      `<p class="muted">What would you like to do?</p><div class="setup-paths">
      <button type="button" data-action="model" data-id="hosted"><strong>Use a hosted vision model</strong><span>Choose from compatible models available through your API key.</span></button>
      <button type="button" data-action="model" data-id="endpoint"><strong>Use an existing segmentation model</strong><span>Connect a deployed U-Net, Hugging Face segmentation endpoint, or another mask service.</span></button>
      <button type="button" data-action="model" data-id="learner"><strong>Train a project model</strong><span>Create a U-Net using your reviewed annotations. No inference endpoint or API key needed.</span></button>
      </div>`,
      async () => false,
    );
    document.querySelector('#action-form button[type="submit"]').hidden = true;
    return;
  }
  const back =
    '<button type="button" class="subtle" data-action="model">← Choose another model type</button>';
  if (path === "hosted") return importHostedModel(ui, back);
  if (path === "custom") {
    modal(
      "Connect a custom vision API",
      back +
        field("Model name", "name", "text", "required") +
        field("Model ID from the provider", "model", "text", "required") +
        selectField(
          "API format",
          "provider",
          '<option value="openai-chat-polygons">Chat Completions</option><option value="openai-polygons">Responses</option><option value="anthropic-polygons">Anthropic Messages</option>',
        ) +
        field("Service URL", "url", "url", "required") +
        credentials() +
        '<p class="muted">For services outside the hosted catalog. The model must accept images and JSON-schema output.</p>',
      async (f) =>
        register(
          f,
          f.get("provider"),
          {
            url: f.get("url").trim(),
            model: f.get("model").trim(),
            timeout: 180,
          },
          f.get("name").trim(),
        ),
      "Connect model",
    );
    bindCredentials("MODEL_API_KEY");
    return;
  }
  if (path === "endpoint") {
    modal(
      "Connect a segmentation service",
      back +
        '<p class="muted">For a trained U-Net, run its weights and preprocessing in a service, then connect it here. Direct weight-file or Hub-repository installation is not yet supported.</p>' +
        field(
          "Model name",
          "name",
          "text",
          "required placeholder='My spleen U-Net'",
        ) +
        selectField(
          "Service type",
          "provider",
          '<option value="http-mask">MONAI Label mask service · 2D / 3D</option><option value="huggingface">Hugging Face segmentation endpoint · 2D</option>',
        ) +
        field(
          "Prediction service URL",
          "url",
          "url",
          "required placeholder='http://localhost:9000/predict'",
        ) +
        labels() +
        '<div id="endpoint-mapping"></div>' +
        credentials() +
        `<details><summary>Advanced options</summary>${field("Model ID from the service (optional)", "model")}</details>`,
      async (f) => {
        const config = { url: f.get("url").trim() };
        if (f.get("model")) config.model = f.get("model").trim();
        if (f.get("provider") === "huggingface") {
          const mapping = {};
          for (const label of state.project.labels) {
            const value = f.get(`map_${label.id}`)?.trim();
            if (value) {
              if (value in mapping)
                throw new Error(
                  "Each service label name must map to one structure.",
                );
              mapping[value] = label.id;
            }
          }
          config.label_map = mapping;
        }
        await register(f, f.get("provider"), config, f.get("name").trim());
      },
      "Connect model",
    );
    bindCredentials("MODEL_API_KEY");
    const form = document.querySelector("#action-form");
    form.elements.auth.value = "none";
    form.elements.auth.dispatchEvent(new Event("change"));
    form.elements.provider.addEventListener("change", () => {
      form.querySelector("#endpoint-mapping").innerHTML =
        form.elements.provider.value === "huggingface"
          ? "<fieldset><legend>Match service labels to project structures</legend>" +
            state.project.labels
              .map((l) =>
                field(
                  `Service label for ${esc(l.name)}`,
                  `map_${l.id}`,
                  "text",
                  `value="${esc(l.name)}"`,
                ),
              )
              .join("") +
            "</fieldset>"
          : "";
    });
    return;
  }
  if (path === "learner") {
    modal(
      "Create a project model",
      back +
        selectField(
          "Model type",
          "recipe",
          [...state.recipes]
            .sort((a, b) =>
              a.id === "monai-unet" ? -1 : b.id === "monai-unet" ? 1 : 0,
            )
            .map(
              (r) =>
                `<option value="${r.id}" ${r.available ? "" : "disabled"}>${esc(r.name)}${r.available ? "" : " · runtime not installed"}</option>`,
            )
            .join(""),
        ) +
        field(
          "Model name",
          "name",
          "text",
          'required maxlength="120" placeholder="e.g. Spleen specialist"',
        ) +
        '<p class="muted">Use this name in chat, for example “train Spleen specialist”.</p>' +
        labels() +
        '<div id="recipe-settings"></div><p class="muted">Save the setup now, then start training when your reviewed data is ready.</p>',
      async (f) => {
        const ids = labelIds(f);
        const config = {};
        const name = f.get("name").trim();
        if (!name) throw new Error("Give this model a name to use in chat.");
        if (
          state.learners.some(
            (l) => l.name.trim().toLowerCase() === name.toLowerCase(),
          )
        )
          throw new Error(
            "A training setup with this name already exists. Choose a different name.",
          );
        const learner = await api(`${prefix}/learners`, "POST", {
          name,
          recipe: f.get("recipe"),
          initial_model_id:
            f.get("recipe") === "vista3d"
              ? state.models.find(
                  (model) => model.preset === "vista3d" && model.read_only,
                )?.id || null
              : null,
          label_ids: ids,
          config,
        });
        state.context.learner_id = learner.id;
        state.page = "models";
        state.modelsTab = "training";
        message(
          `Created ${name}. Accept annotation reviews, then say “train this model” or use Start training.`,
        );
      },
      "Create model",
    );
    const form = document.querySelector("#action-form");
    function recipeChanged() {
      const recipe = state.recipes.find(
        (item) => item.id === form.elements.recipe.value,
      );
      form.querySelector('button[type="submit"]').disabled = !recipe?.available;
      form.querySelector("#recipe-settings").innerHTML =
        form.elements.recipe.value === "monai-unet"
          ? '<p class="muted">Train a U-Net from scratch on 2D images or 3D scans. The image format is detected automatically. Start with the recommended settings; adjust them when you start training.</p>'
          : form.elements.recipe.value === "vista3d"
            ? '<p class="muted">Create a separate CT model based on VISTA3D. Fine-tuning uses the selected organs; the shared base stays read-only. A compatible GPU is required by the default recipe.</p>'
            : `<p class="muted">${esc(recipe?.description || "Install a training runtime on the server to create a setup.")}</p>`;
    }
    form.elements.recipe.addEventListener("change", recipeChanged);
    recipeChanged();
  }
}
