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

import {jobFinished} from "./ui.js";

export function exportProgress(initial, {api, modal}) {
  modal("Dataset export", '<div id="export-progress" aria-live="polite"></div>', async () => true, "Close");
  const host = document.querySelector("#export-progress");
  const dialog = host.closest("dialog");
  let timer;
  const alive = () => host.isConnected && dialog.open;
  const render = job => {
    host.replaceChildren();
    const note = document.createElement("p"); note.textContent = job.error || job.progress_message || "Preparing ZIP…";
    host.append(note);
    if (job.status === "succeeded") {
      const ready = document.createElement("p"); ready.textContent = `${job.result.asset_count} samples · ${job.result.annotation_count} saved annotation versions`;
      const link = document.createElement("a"); link.className = "download-link"; link.textContent = "Download ZIP";
      link.href = `/api/jobs/${job.id}/export`; link.download = job.result.filename; host.append(ready, link);
    } else if (!jobFinished(job)) {
      const progress = document.createElement("progress"); progress.max = 1; progress.value = job.progress;
      progress.setAttribute("aria-label", "Dataset export progress"); host.append(progress);
      const background = document.createElement("p"); background.className = "muted";
      background.textContent = "You can close this window. The download will be available in Activity."; host.append(background);
    } else note.className = "form-error";
  };
  async function poll() {
    try {
      const job = await api(`/jobs/${initial.id}`);
      if (!alive()) return;
      render(job);
      if (!jobFinished(job)) timer = setTimeout(poll, 700);
    } catch (error) { if (alive()) host.textContent = error.message; }
  }
  dialog.addEventListener("close", () => clearTimeout(timer), {once: true});
  render(initial);
  if (!jobFinished(initial)) timer = setTimeout(poll, 300);
}

export function exportDataset({state, api, modal}) {
  const ids = [...state.selectedFiles];
  if (!ids.length) throw new Error("Select samples to export first.");
  const project = state.project.id;
  modal("Export selected samples", `<p>${ids.length} selected samples with original images and saved labels.</p>
    <label>Annotation versions<select name="annotations"><option value="latest">Latest saved version</option><option value="all">All saved versions</option></select></label>
    <p class="muted">Includes a manifest with classes, geometry, grouping and review status. Unsaved viewer drafts are excluded.</p>`,
    async form => {
      const job = await api(`/projects/${project}/exports`, "POST", {asset_ids: ids, annotations: form.get("annotations")});
      // The modal helper finishes its submission before the progress window opens.
      setTimeout(() => exportProgress(job, {api, modal}), 0);
      return false;
    }, "Prepare ZIP");
}
