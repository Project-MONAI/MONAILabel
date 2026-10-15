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

import { escapeHTML as esc } from "./ui.js";

const examples = [["nnunet-v2", "nnU-Net training"], ["vista3d", "VISTA3D fine-tuning"]];

function recordedFactsHTML(example) {
  const score = example.mean_dice == null
    ? "No held-out score is available."
    : `Recorded mean Dice: ${Number(example.mean_dice).toFixed(3)}.`;
  return `<p><strong>Recorded run — completed before your visit.</strong> Your current corrections were not used in this run.</p>
    <dl class="details-list"><dt>Dataset</dt><dd>${esc(example.dataset)}</dd>
    <dt>Completed</dt><dd>${esc(example.completed_at)}</dd>
    <dt>Training / evaluation cases</dt><dd>${example.training_cases} / ${example.evaluation_cases}</dd>
    <dt>Run settings</dt><dd>${example.epochs} epochs × ${example.steps_per_epoch} steps per epoch</dd>
    <dt>Recorded training duration</dt><dd>${Math.round(example.duration_seconds)} seconds</dd></dl>
    <p>${score}</p>
    ${example.final_loss == null ? "" : `<p>Recorded final training loss: ${Number(example.final_loss).toFixed(4)}.</p>`}`;
}

function recordedDetailsHTML(example) {
  return `<p>${esc(example.name)}</p>
    <label>Recorded training log<textarea aria-label="Recorded training log" class="code training-log" readonly rows="10">${esc(example.log)}</textarea></label>
    <h4>Provenance and limitations</h4>
    <ul>${example.caveats.map((note) => `<li>${esc(note)}</li>`).join("")}</ul>
    <p>Checkpoint SHA-256</p><p class="code">${esc(example.checkpoint_sha256)}</p>
    <p>Source base revision</p><p class="code">${esc(example.source_revision)}</p>`;
}

export function recordedTrainingHTML(example) {
  return `<h3>${esc(example.name)}</h3>${recordedFactsHTML(example)}${recordedDetailsHTML(example)}`;
}

export function recordedTrainingCardsHTML() {
  return `<section aria-labelledby="recorded-training-heading" data-recorded-examples>
    <div class="section-heading"><div><h2 id="recorded-training-heading">Recorded examples</h2>
    <p class="muted">Training preview — no training will run. These public-data runs were completed before your visit, not on your corrections. They demonstrate the workflow, not clinical accuracy.</p></div></div>
    <div class="model-grid recorded-training-grid">${examples.map(([recipe, name]) => `
      <article class="card" data-recorded-recipe="${recipe}" aria-labelledby="recorded-${recipe}-title">
        <div class="model-top"><h3 id="recorded-${recipe}-title">${name}</h3><span class="badge">Recorded</span></div>
        <div data-recorded-body aria-live="polite"><p>Loading recorded example…</p></div>
      </article>`).join("")}</div></section>`;
}

export function createTrainingExampleLoader(api) {
  // The two public evidence records are bounded, shared and read-only. Reuse
  // requests during dashboard refreshes; clear them on leaving authentication.
  const cache = new Map();
  function read(recipe) {
    let entry = cache.get(recipe);
    if (!entry || entry.expires <= Date.now()) {
      entry = { expires: Date.now() + 60000, promise: api(`/demo-access/training/${recipe}`) };
      cache.set(recipe, entry);
      entry.promise.catch(() => { if (cache.get(recipe) === entry) cache.delete(recipe); });
    }
    return entry.promise;
  }
  async function fill(card) {
    const output = card.querySelector("[data-recorded-body]");
    try {
      const example = await read(card.dataset.recordedRecipe);
      if (!card.isConnected) return;
      output.innerHTML = `${recordedFactsHTML(example)}
        <details><summary>View recorded log and run details</summary>${recordedDetailsHTML(example)}</details>`;
    } catch (error) {
      if (!card.isConnected) return;
      output.replaceChildren();
      const message = document.createElement("p");
      message.textContent = error.message;
      message.setAttribute("role", "status");
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "Retry loading example";
      retry.onclick = () => {
        output.textContent = "Loading recorded example…";
        void fill(card);
      };
      output.append(message, retry);
    }
  }
  return {
    clear: () => cache.clear(),
    load: (root) => Promise.all([...root.querySelectorAll("[data-recorded-recipe]")].map(fill)),
  };
}

export async function showTrainingPreview({ api, modal }, learner = null) {
  const info = await api("/demo-access/training");
  modal(
    learner ? `Preview training: ${learner.name}` : "Recorded training examples",
    `<section data-training-preview>
      <p role="status"><strong>Training preview — no training will run.</strong></p>
      <p>Live training is disabled to keep this shared demo responsive. Recorded runs used public reference data, not your corrections. No job will be queued.</p>
      ${learner ? `<p>Selected setup: ${esc(learner.name)} (${esc(learner.recipe)}). This walkthrough does not validate training eligibility or save a snapshot.</p>` : ""}
      <details><summary>What would happen in a full installation</summary>
      <ol>${info.steps.map((step) => `<li>${esc(step)}</li>`).join("")}</ol></details>
      <label>Recorded example<select data-training-example aria-label="Recorded example">
        <option value="">Choose a completed example</option>
        <option value="nnunet-v2">nnU-Net training</option>
        <option value="vista3d">VISTA3D fine-tuning</option>
      </select></label>
      <div data-recorded-training aria-live="polite"></div>
    </section>`,
    async () => true,
    "Close",
  );
  const host = document.querySelector("[data-training-preview]");
  const select = host.querySelector("[data-training-example]");
  const output = host.querySelector("[data-recorded-training]");
  let generation = 0;
  select.onchange = async () => {
    const current = ++generation;
    output.textContent = select.value ? "Loading recorded example…" : "";
    if (!select.value) return;
    try {
      const result = await api(`/demo-access/training/${select.value}`);
      if (host.isConnected && current === generation)
        output.innerHTML = recordedTrainingHTML(result);
    } catch (error) {
      if (host.isConnected && current === generation) output.textContent = error.message;
    }
  };
}
