import { escapeHTML } from "./ui.js";

const numbers = {
  epochs: ["Epochs", 1, 1000, 1],
  steps_per_epoch: ["Training steps per epoch", 1, 1000, 1],
  batch_size: ["Batch size", 1, 32, 1],
  learning_rate: ["Learning rate", 0.000000001, 0.1, "any"],
  weight_decay: ["Weight decay", 0, 1, "any"],
  patch_size: ["Training crop size", 16, 192, 8],
  seed: ["Random seed", 0, 4294967295, 1],
  spacing: ["Voxel spacing (mm)", 0.5, 4, "any"],
};
const hints = {
  epochs:
    "How many groups of training steps to run. Patches are sampled; an epoch is not a full pass through every image.",
  steps_per_epoch:
    "Number of weight updates in each epoch. Each step uses one batch of sampled patches.",
  batch_size:
    "Patches used for each weight update. Larger batches use more memory.",
  learning_rate:
    "How much each update can change the model. Start with the recommended value.",
  weight_decay:
    "Regularization that discourages large weights. Zero turns it off.",
  patch_size:
    "Width of each square or cube used for training, in pixels or voxels.",
  seed: "Controls random choices to help make runs repeatable.",
  spacing:
    "Resample 3D scans to this spacing. Leave blank to keep the original spacing.",
};
export const trainingSettingLabel = (key) =>
  numbers[key]?.[0] ||
  {
    device: "Run training on",
    intensity_window: "Image intensity range",
    spatial_dims: "Image dimensions",
    in_channels: "Image channels",
    channels: "Network layer sizes",
    strides: "Network downsampling",
    bundle: "Model bundle",
    bundle_version: "Bundle version",
    vista_label_ids: "Original structure IDs",
    label_mapping: "Structure mapping",
  }[key] ||
  key.replaceAll("_", " ");
export function trainingSettings(learner) {
  const config = learner.config;
  const fields = Object.entries(numbers)
    .filter(
      ([key]) =>
        key in config && !(key === "spacing" && config.spatial_dims === 2),
    )
    .map(([key, [label, min, max, step]]) => {
      if (learner.recipe === "vista3d") {
        if (key === "epochs") max = 100;
        if (key === "learning_rate") max = 0.01;
        if (key === "patch_size") {
          min = 32;
          step = 16;
        }
      } else if (key === "patch_size") max = 128;
      const hint =
        learner.recipe === "vista3d" && key === "batch_size"
          ? "Patches used for each weight update. VISTA3D processes them one at a time and averages their gradients."
          : hints[key];
      return `<div><label>${label}<input name="run_${key}" aria-describedby="run-${key}-hint" type="number" min="${min}" max="${max}" step="${step}" value="${config[key] ?? ""}" ${key === "spacing" ? "" : "required"}></label><small id="run-${key}-hint">${hint}</small></div>`;
    })
    .join("");
  if (!fields) return "";
  return `<details><summary>Training settings (recommended)</summary><p class="muted">Changes apply to this run only.</p><div class="form-grid">${fields}${
    "device" in config
      ? `<label>Run training on<select name="run_device">${[
          "auto",
          "cpu",
          "cuda",
        ]
          .filter((d) => learner.recipe !== "vista3d" || d !== "auto")
          .map(
            (d) =>
              `<option value="${d}" ${config.device === d ? "selected" : ""}>${{ auto: "Automatic", cpu: "CPU", cuda: "GPU (CUDA)" }[d]}</option>`,
          )
          .join("")}</select></label>`
      : ""
  }${"intensity_window" in config && config.spatial_dims !== 2 ? `<div><label>Image intensity range<input name="run_intensity_window" aria-describedby="run-range-hint" placeholder="Automatic" value="${escapeHTML(config.intensity_window?.join(", ") || "")}"></label><small id="run-range-hint">Optional: minimum, maximum (for example, -200, 300). Leave blank for automatic scaling.</small></div>` : ""}</div><p class="muted" data-training-budget role="status"></p></details>`;
}

export function bindTrainingSettings(form) {
  const output = form.querySelector("[data-training-budget]");
  if (!output) return;
  const update = () => {
    const value = (key) => Number(form.elements[`run_${key}`]?.value || 0);
    const steps = value("epochs") * value("steps_per_epoch");
    const patches = steps * value("batch_size");
    output.textContent =
      steps && patches
        ? `${steps.toLocaleString()} weight updates · ${patches.toLocaleString()} sampled patches`
        : "Enter epochs, training steps and batch size to see the run size.";
  };
  form.addEventListener("input", update);
  update();
}
export function trainingOverrides(form, learner) {
  const result = {};
  for (const [key, raw] of form) {
    if (!key.startsWith("run_")) continue;
    const name = key.slice(4);
    let value = raw;
    if (name === "intensity_window") {
      value = raw.trim() ? raw.split(",").map(Number) : null;
      if (value && (value.length !== 2 || !value.every(Number.isFinite)))
        throw new Error("Use two numbers for the intensity window: low, high.");
    } else if (name !== "device") {
      value = raw === "" ? null : Number(raw);
    }
    if (JSON.stringify(value) !== JSON.stringify(learner.config[name]))
      result[name] = value;
  }
  return result;
}
