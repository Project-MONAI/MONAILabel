/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import {
  evaluationVersionOptions,
  evaluationVersionName,
} from "./evaluation-sets.js";
import { actionLabel, icon } from "./icons.js";
import { trainingSettingLabel } from "./training-settings.js";
import { escapeHTML, badge, button, targetNames } from "./ui.js";
import { targetSummary } from "./model-targets.js";
import { reviewItems } from "./review-items.js";

export function modelLibrary(state, canManage, latestDecision) {
  const tab = state.modelsTab;
  const activeSets = state.evaluationSets.filter((s) => !s.archived);
  const currentVersion = state.evaluationVersions.find(
    (v) => v.id === state.context.evaluation_version_id,
  );
  const selectedSet =
    activeSets.find(
      (s) =>
        s.id === state.context.evaluation_set_id ||
        (!state.context.evaluation_set_id &&
          s.id === currentVersion?.evaluation_set_id),
    ) || (activeSets.length === 1 ? activeSets[0] : null);
  const evaluationOptions = selectedSet
    ? evaluationVersionOptions(
        { ...state, evaluationSets: [selectedSet] },
        state.context.evaluation_version_id,
      )
    : "";
  const tabs = `<div class="tabs" aria-label="Model sections">${[
    ["annotation", "Annotation"],
    ["training", "Training"],
    ["evaluation", "Evaluation"],
  ]
    .map(
      ([id, label]) =>
        `<button data-action="model-tab" data-id="${id}" aria-pressed="${tab === id}" class="${tab === id ? "active" : ""}">${label}</button>`,
    )
    .join("")}</div>`;
  const names = (ids) =>
    state.project.labels
      .filter((l) => l.id && ids.includes(l.id))
      .map((l) => l.name)
      .join(", ");
  const trained = (model) =>
    Boolean(model.snapshot_id || model.learner_id || model.mode);
  const predefined = state.models.filter((m) => m.preset && !trained(m));
  const category = (model) =>
    ["vista3d", "totalsegmentator-ct", "totalsegmentator-mr"].includes(
      model.provider,
    )
      ? "radiology"
      : Boolean(model.interaction)
        ? "interactive"
        : [
              "openai-polygons",
              "openai-chat-polygons",
              "anthropic-polygons",
            ].includes(model.provider)
          ? "vision"
          : "other";
  const origin = (model) => {
    const parent = state.models.find((item) => item.id === model.parent_id);
    if (model.mode === "scratch") return "Trained from scratch";
    if (model.mode === "fine_tune")
      return parent ? `Fine-tuned from ${parent.name}` : "Fine-tuned model";
    if (model.mode === "continue")
      return parent ? `Continued from ${parent.name}` : "Continued training";
    if (trained(model)) return "Trained in this project";
    if (
      [
        "openai-polygons",
        "openai-chat-polygons",
        "anthropic-polygons",
      ].includes(model.provider)
    )
      return "Vision-language model";
    return model.preset ? "Pretrained model" : "Added to this project";
  };
  function card(m) {
    const provider = providerName(m);
    const trainingSetup = state.learners.find(
      (learner) => learner.id === m.learner_id,
    );
    const isDefault = state.project.annotation_model_id === m.id;
    const spatial = Boolean(m.interaction);
    const total = ["totalsegmentator-ct", "totalsegmentator-mr"].includes(
      m.provider,
    );
    const logo = !trained(m) && m.preset === "vista3d" ? "monai.ico" : null;
    const symbol = trained(m)
      ? "network"
      : m.provider === "totalsegmentator-mr"
        ? "magnet"
        : category(m) === "radiology"
          ? "scan"
          : spatial
            ? "segment"
            : category(m) === "vision"
              ? "vision"
              : "server";
    const scope = total
      ? `${m.provider === "totalsegmentator-ct" ? "3D CT" : "3D MRI"} · 3 mm · local`
      : spatial
        ? m.interaction?.volume_only
          ? "3D volume · local"
          : "2D / selected slice · local"
        : ["monai-unet", "nnunet-v2", "vista3d"].includes(m.provider)
          ? m.provider === "vista3d"
            ? "3D CT · local"
            : m.provider === "nnunet-v2"
              ? `3D ${m.config.modality} · nnU-Net v2`
              : `${m.config.spatial_dims || 3}D U-Net`
          : [
                "openai-polygons",
                "openai-chat-polygons",
                "anthropic-polygons",
                "huggingface",
              ].includes(m.provider)
            ? "2D / selected slice"
            : m.provider === "pixel-gaussian" || m.provider === "threshold"
              ? "Demo baseline"
              : "2D / 3D mask service";
    const derive =
      m.read_only &&
      ["vista3d", "totalsegmentator-ct", "totalsegmentator-mr"].includes(
        m.provider,
      ) &&
      canManage();
    return `<article class="card" data-model-id="${escapeHTML(m.id)}">
      <div class="model-top"><span class="model-symbol${logo ? " model-symbol-image" : ""}" aria-hidden="true">${logo ? `<img src="/static/model-icons/${logo}" alt="" width="30" height="30">` : icon(symbol)}</span><h3>${escapeHTML(m.name)}</h3>${isDefault ? badge("Default") : ""}</div>
      <p class="model-provenance">${escapeHTML(origin(m))}</p>
      <p class="model-provenance">Provider: ${escapeHTML(provider)}${m.connection_mode ? ` · ${m.connection_mode === "manual" ? "Selected for this project" : "Automatic"}` : ""}</p>
      <div class="model-meta">${badge(total ? (m.provider === "totalsegmentator-ct" ? "117 structures" : "50 structures") : spatial ? "Box / point prompts" : m.read_only && m.provider === "vista3d" ? "CT anatomy" : ["openai-polygons", "openai-chat-polygons", "anthropic-polygons"].includes(m.provider) ? "Prompt-defined labels" : names(m.label_ids))}${badge(scope)}${m.unreviewed_training ? badge("Trained on unreviewed predictions") : ""}</div>
      <div class="model-card-footer">
        <details><summary>Model details</summary><div class="model-detail-body">
          ${targetSummary(state, m)}
          ${total ? '<p class="muted">Apache-2.0 · Local inference and fine-tuning. Weights download on first use.</p>' : ""}
          ${m.provider === "vista3d" ? '<p class="muted"><a href="https://huggingface.co/MONAI/vista3d/blob/c6dbe159632a4767696e09f91d74d729b82e73e6/LICENSE" target="_blank" rel="noopener noreferrer">Weight license</a>: noncommercial research/evaluation only, including fine-tuned weights.</p>' : ""}
          ${m.provider === "nninteractive" ? '<p class="muted"><a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="noopener noreferrer">Weights: CC BY-NC-SA 4.0</a> · Noncommercial use, attribution and share-alike. Downloads on first use.</p>' : ""}
          ${m.provider === "medsam2" ? '<p class="muted"><a href="https://huggingface.co/wanglab/MedSAM2/blob/e4a6f35edd7e091619cbc0750f462f1574e23955/README.md#license" target="_blank" rel="noopener noreferrer">Weight terms</a>: research and education only.</p>' : ""}
          <p class="code">${escapeHTML(m.config.model || m.provider)}</p>
          <p class="muted">${escapeHTML(m.config.url || (m.read_only ? "Pinned shared weights · reused locally" : "Local project checkpoint"))}</p>
          ${evaluationDetails(state, m)}
          <div class="model-detail-actions">${!isDefault && canManage() ? button("Make default", "default-model", m.id, "model-text-action") : ""}${spatial ? "" : button("Use as evaluation baseline", "baseline", m.id, "model-text-action")}</div>
        </div></details>
        ${canManage() && m.preset && m.connection_mode ? button("Change provider", "model-provider", m.id, "model-text-action") : ""}
        ${canManage() && !m.preset ? `<div class="model-detail-actions">${trainingSetup ? button("Manage model", "manage-training-model", trainingSetup.id, "model-text-action") : button("Rename", "rename-model", m.id, "model-text-action") + button("Delete", "delete-model", m.id, "model-text-action")}</div>` : ""}
        ${derive ? button('Create project model <span aria-hidden="true">→</span>', "derive-model", m.id, "model-text-action model-derive") : ""}
      </div>
    </article>`;
  }
  function group(id, title, description, list) {
    return list.length
      ? `<section class="model-group" aria-labelledby="model-group-${id}"><header class="model-group-header"><h2 id="model-group-${id}">${title}<span class="model-group-count" aria-label="${list.length} models">${list.length}</span></h2><p>${description}</p></header><div class="model-grid">${list.map(card).join("")}</div></section>`
      : "";
  }
  const accepted = (split) =>
    new Set(
      reviewItems(state)
        .filter(
          (a) => a.split === split && latestDecision(a)?.verdict === "accepted",
        )
        .map((a) => a.source_id || a.id),
    ).size;
  return `<div class="section-heading"><div><p class="muted">Manage annotation models and training. Choose a model in chat or your viewer when annotating.</p></div><div class="toolbar">${canManage() ? button("API keys", "credential") + button("Add model", "model", "", "primary") : ""}</div></div>
    ${tabs}
    ${
      tab === "annotation"
        ? [
            group(
              "project",
              "Project models",
              "Trained or fine-tuned in this project.",
              state.models.filter(trained),
            ),
            group(
              "added",
              "Added models",
              "Models and services connected to this project.",
              state.models.filter((m) => !trained(m) && !m.preset),
            ),
            group(
              "radiology",
              "Radiology segmentation",
              "Predefined models for CT and MRI anatomy.",
              predefined.filter((m) => category(m) === "radiology"),
            ),
            group(
              "interactive",
              "Interactive segmentation",
              "Guide segmentation with boxes or points.",
              predefined.filter((m) => category(m) === "interactive"),
            ),
            group(
              "vision",
              "Vision-language models",
              "Use text prompts to annotate images, slices and video frames.",
              predefined.filter((m) => category(m) === "vision"),
            ),
            group(
              "other",
              "Other predefined models",
              "Additional annotation models.",
              predefined.filter((m) => category(m) === "other"),
            ),
          ].join("")
        : ""
    }
    ${tab === "annotation" && !state.models.length ? '<div class="empty">No annotation models connected. Add a hosted model or an existing segmentation service to begin.</div>' : ""}
    ${
      tab === "training"
        ? `<div class="section-heading"><div><h2>Training setups</h2><p class="muted">${accepted("train") + accepted("pool")} source files with accepted coverage · each model has its own split</p></div>${canManage() ? button("Create model", "model", "learner") : ""}</div>
    <div class="model-grid">${
      state.learners
        .map((l) => {
          const versions = state.models.filter((m) => m.learner_id === l.id);
          const completed = [...state.jobs]
            .reverse()
            .find(
              (j) =>
                j.kind === "train" &&
                j.request.learner_id === l.id &&
                j.status === "succeeded",
            );
          const running = state.jobs.some(
            (j) =>
              j.request.learner_id === l.id &&
              ["queued", "running"].includes(j.status),
          );
          return `<article class="card" data-learner-id="${l.id}"><div class="model-top"><h3>${escapeHTML(l.name)}</h3>${badge(running ? "training" : versions.length ? `${versions.length} trained versions` : "Not trained")}</div><p>${escapeHTML(state.recipes.find((r) => r.id === l.recipe)?.name || l.recipe)} · ${escapeHTML(names(l.label_ids) || (l.inherit_targets ? "Choose organs when training" : "No targets"))}</p><p class="muted">${["monai-unet", "nnunet-v2", "vista3d", "totalsegmentator-ct", "totalsegmentator-mr"].includes(l.recipe) ? (l.recipe === "monai-unet" && l.config.spatial_dims === 2 ? "Local neural network · 2D images" : "Local neural network · scalar volumes") : "Workflow demonstration baseline"}</p><div class="toolbar">${canManage() && !running ? button("Start training", "start-training", l.id, "primary") : ""}${state.modelSplits.some((s) => s.learner_id === l.id) ? button("Evaluation set", "model-split", l.id) : ""}${completed ? button("Results", "job-details", completed.id) : ""}${canManage() && !running ? button("Rename", "rename-learner", l.id) + button("Delete", "delete-learner", l.id) : ""}</div><details><summary>Recommended training settings</summary><p class="muted">Override these when you start a training run.</p><dl class="details-list">${Object.entries(
            {
              ...state.recipes.find((r) => r.id === l.recipe)?.default_config,
              ...l.config,
            },
          )
            .filter(([key]) => key !== "device")
            .map(
              ([key, value]) =>
                `<dt>${escapeHTML(trainingSettingLabel(key))}</dt><dd>${escapeHTML(typeof value === "object" ? JSON.stringify(value) : value)}</dd>`,
            )
            .join("")}</dl></details></article>`;
        })
        .join("") ||
      '<div class="empty">Try chatting: “create u-net based segmentation model for spleen”. Use a structure defined in your project.</div>'
    }</div>`
        : ""
    }
    ${
      tab === "evaluation"
        ? `<section class="card"><h3>Compare models</h3>${activeSets.length ? `<label>Evaluation set<select data-context="evaluation_set_id" aria-label="Evaluation set"><option value="" ${selectedSet ? "" : "selected"} disabled>Choose an evaluation set</option>${activeSets.map((s) => `<option value="${s.id}" ${selectedSet?.id === s.id ? "selected" : ""}>${escapeHTML(s.name)}</option>`).join("")}</select></label>` : '<p class="muted">Uses the project’s accepted evaluation cases.</p>'}${selectedSet && state.evaluationVersions.filter((v) => v.evaluation_set_id === selectedSet.id).length > 1 ? `<details ${state.context.evaluation_version_id ? "open" : ""}><summary>Evaluation options</summary><label>Reference version<select data-context="evaluation_version_id" aria-label="Reference version"><option value="">Latest saved version</option>${evaluationOptions}</select></label></details>` : ""}${button("Manage evaluation sets", "evaluation-sets")}<p class="muted">Choose two models that support the same targets and image type. Compare them on reviewed evaluation samples kept separate from training.</p><div class="comparison-selectors">${[
            ["model_id", "Model to evaluate"],
            ["baseline_id", "Compare against"],
          ]
            .map(
              ([key, label]) =>
                `<label>${label}<select data-context="${key}" aria-label="${label}"><option value="">Select a model</option>${state.models
                  .filter((m) => !m.interaction)
                  .map(
                    (m) =>
                      `<option value="${m.id}" ${state.context[key] === m.id ? "selected" : ""}>${escapeHTML(m.name)}</option>`,
                  )
                  .join("")}</select></label>`,
            )
            .join(
              "",
            )}</div>${canManage() ? `<div class="toolbar"><button type="button" data-action="evaluate" ${(!activeSets.length || selectedSet) && state.context.model_id && state.context.baseline_id && state.context.model_id !== state.context.baseline_id ? "" : "disabled"}>${actionLabel("Compare models", "compare")}</button></div>` : ""}</section>${
            state.models
              .filter((m) =>
                state.evaluations.some((e) => e.candidate_id === m.id),
              )
              .map(
                (m) =>
                  `<article class="card"><h3>${escapeHTML(m.name)}</h3>${evaluationDetails(state, m)}${canManage() && [...state.evaluations].reverse().find((e) => e.candidate_id === m.id)?.eligible_labels.length ? button("Promote eligible targets", "promote", [...state.evaluations].reverse().find((e) => e.candidate_id === m.id).id) : ""}</article>`,
              )
              .join("") ||
            '<div class="empty">No comparisons yet. Choose evaluation data and compare two models.</div>'
          }`
        : ""
    }
    `;
}
export function providerName(model) {
  if (!model.config.url) return "Local";
  try {
    const host = new URL(model.config.url).hostname;
    if (host === "inference-api.nvidia.com") {
      const route = String(model.config.model || "")
        .split("/")
        .slice(0, -1)
        .join("/");
      return route ? `NVIDIA · ${route}` : "NVIDIA";
    }
    return (
      {
        "api.openai.com": "OpenAI",
        "api.anthropic.com": "Anthropic (Claude)",
        "generativelanguage.googleapis.com": "Google (Gemini)",
      }[host] || host
    );
  } catch {
    return "Custom service";
  }
}
function evaluationDetails(state, model) {
  if (Boolean(model.interaction)) return "";
  const result = [...state.evaluations]
    .reverse()
    .find((e) => e.candidate_id === model.id);
  if (!result) return '<p class="muted">No held-out evaluation yet.</p>';
  const baseline = state.models.find((m) => m.id === result.baseline_id);
  return `<p><strong>Held-out Dice</strong> · 0–1, higher is better</p>
    <table><thead><tr><th>Target</th><th>This model</th><th>Baseline</th></tr></thead><tbody>${Object.entries(
      result.candidate.per_class,
    )
      .map(
        ([id, score]) =>
          `<tr><td>${escapeHTML(state.project.labels.find((l) => l.id === Number(id))?.name || id)}</td><td>${Number(score).toFixed(3)}</td><td>${Number(result.baseline.per_class[id]).toFixed(3)}</td></tr>`,
      )
      .join("")}</tbody></table>
    <p class="muted">${escapeHTML(evaluationVersionName(state, result.evaluation_version_id))} · ${result.validation_assets.length} held-out cases · Baseline: ${escapeHTML(baseline?.name || "Unavailable")}<br>${result.eligible_labels.length ? "Targets eligible for promotion: " + escapeHTML(targetNames(state.project, result.eligible_labels)) : "No target met the promotion criteria."}</p>`;
}
