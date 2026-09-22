import { trainingSettingLabel } from "./training-settings.js";
import { escapeHTML as esc, badge, jobFinished } from "./ui.js";

function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function resultHTML(report, state) {
  if (!report) return "";
  const version = state.evaluationVersions.find(
    (v) => v.id === report.evaluation_version_id,
  );
  const set = state.evaluationSets.find(
    (s) => s.id === version?.evaluation_set_id,
  );
  const reference = set
    ? `${set.name} · v${version.number}`
    : report.model_split_id
      ? `Model validation · split v${report.model_split_version}`
      : "This run’s evaluation cases";
  const loss =
    report.final_loss == null
      ? ""
      : `<p class="muted">Training loss: ${Number(report.initial_loss).toFixed(4)} → ${Number(report.final_loss).toFixed(4)}</p>`;
  if (!report.metrics)
    return `<p role="status">${esc(report.evaluation_requested === false ? "Trained without evaluation. No held-out score is available." : report.error || "Evaluation is unavailable.")}</p>${loss}`;
  return `<h3>Mean Dice: ${Number(report.metrics.mean_dice).toFixed(3)}</h3><p class="muted">${report.validation_assets.length} validation case${report.validation_assets.length === 1 ? "" : "s"} · ${esc(reference)}</p><table><thead><tr><th>Structure</th><th>Dice</th><th>IoU</th></tr></thead><tbody>${report.labels.map((l) => `<tr><td>${esc(l.name)}</td><td>${Number(report.metrics.per_class[l.id]).toFixed(3)}</td><td>${Number(report.per_class_iou[l.id]).toFixed(3)}</td></tr>`).join("")}</tbody></table><p class="muted">Scores range from 0 to 1; higher is better. Each structure’s score combines its voxels or pixels across the evaluation cases. Mean Dice averages the foreground structures.</p>${loss}<button type="button" data-download-report>Download report</button>`;
}

function comparisonHTML(report, state) {
  const name = (id) =>
    state.models.find((m) => m.id === id)?.name || "Selected model";
  const structures = Object.keys(report.candidate.per_class);
  return `<h3>Model comparison</h3><p class="muted">${report.validation_assets.length} held-out evaluation cases · Dice scores</p><table><thead><tr><th>Structure</th><th>${esc(name(report.candidate_id))}</th><th>${esc(name(report.baseline_id))}</th></tr></thead><tbody><tr><th>Mean Dice</th><td>${Number(report.candidate.mean_dice).toFixed(3)}</td><td>${Number(report.baseline.mean_dice).toFixed(3)}</td></tr>${structures.map((id) => `<tr><td>${esc(state.project.labels.find((l) => l.id === Number(id))?.name || "Structure " + id)}</td><td>${Number(report.candidate.per_class[id]).toFixed(3)}</td><td>${Number(report.baseline.per_class[id]).toFixed(3)}</td></tr>`).join("")}</tbody></table><p class="muted">Scores range from 0 to 1; higher is better. Both models use the same fixed evaluation references.</p><button type="button" data-download-report>Download report</button>`;
}

export function trainingResults(initial, { state, api, modal, canManage }) {
  const comparing = initial.kind === "evaluate";
  const training = initial.kind === "train";
  const batch = initial.kind === "batch_annotate";
  const title = batch
    ? "Batch segmentation"
    : training
      ? "Training"
      : "Evaluation";
  const modelName = (id) =>
    state.models.find((m) => m.id === id)?.name || "Selected model";
  const settings = batch
    ? `<dt>Model</dt><dd>${esc(modelName(initial.request.model_id))}</dd><dt>Images</dt><dd>${initial.request.asset_ids.length}</dd><dt>Structures</dt><dd>${esc((initial.request.label_ids || []).map((id) => state.project.labels.find((l) => l.id === id)?.name || id).join(", "))}</dd><dt>After segmentation</dt><dd>${initial.request.submit_for_review ? "Submit for pending review" : "Save proposals"}</dd>`
    : comparing
      ? `<dt>Model to evaluate</dt><dd>${esc(modelName(initial.request.candidate_id))}</dd><dt>Compare against</dt><dd>${esc(modelName(initial.request.baseline_id))}</dd>`
      : training
        ? `<dt>Model</dt><dd>${esc(initial.request.name || "Project model")}</dd><dt>How training started</dt><dd>${esc({ scratch: "New model", fine_tune: "Fine-tune", continue: "Continue training" }[initial.request.mode] || "Training run")}</dd>`
        : "<dt>Task</dt><dd>Evaluate the saved training checkpoint</dd>";
  modal(
    `${title} ${initial.status === "succeeded" ? "results" : "logs"}`,
    `<section data-training-results><details><summary>Run settings</summary><dl class="details-list">${settings}${Object.entries(
      initial.request.config || {},
    )
      .map(
        ([key, value]) =>
          `<dt>${esc(trainingSettingLabel(key))}</dt><dd>${esc(typeof value === "object" ? JSON.stringify(value) : value)}</dd>`,
      )
      .join(
        "",
      )}</dl></details><div data-run-status role="status"></div><div data-report></div><p data-report-note class="muted"></p><button type="button" data-run-report hidden>Evaluate model</button><p class="form-error" data-results-error role="alert"></p><section class="training-log-panel"><h3>${title} log</h3><div class="training-log-controls"><label>Recent lines<select data-log-limit aria-label="Recent lines"><option value="100">100</option><option value="500">500</option><option value="1000" selected>1,000</option></select></label><label class="checkbox-row"><input type="checkbox" data-follow-log checked>Follow latest</label><a class="download-link" href="/api/jobs/${initial.id}/logs/download" download>Download full log</a></div><p data-log-note class="muted"></p><textarea class="code training-log" data-training-log readonly spellcheck="false" wrap="off" rows="16" aria-label="${title} log"></textarea></section></section>`,
    async () => true,
    "Close",
  );
  const host = document.querySelector("[data-training-results]");
  const dialog = host.closest("dialog");
  const alive = () => host.isConnected && dialog.open;
  const output = host.querySelector("[data-training-log]");
  const error = host.querySelector("[data-results-error]");
  const run = host.querySelector("[data-run-report]");
  let cursor = 0,
    entries = [],
    reportId = "",
    reportJob = null,
    timer = null,
    busy = false;
  const count = host.querySelector("[data-log-limit]");
  let refreshPending = false;
  count.onchange = () => {
    cursor = 0;
    entries = [];
    output.value = "";
    clearTimeout(timer);
    if (busy) refreshPending = true;
    else void poll();
  };
  const stop = () => clearTimeout(timer);
  dialog.addEventListener("close", stop, { once: true });
  async function poll() {
    if (!alive() || busy) return;
    busy = true;
    const limit = Number(count.value);
    try {
      const [job, logs, runningReport] = await Promise.all([
        api(`/jobs/${initial.id}`),
        api(`/jobs/${initial.id}/logs?after=${cursor}&limit=${limit}`),
        reportJob ? api(`/jobs/${reportJob}`) : null,
      ]);
      if (!alive()) return;
      let report = batch
        ? null
        : comparing
          ? job.result.evaluation_id
            ? await api(`/evaluations/${job.result.evaluation_id}`)
            : null
          : await api(`/jobs/${initial.id}/training-report`);
      if (runningReport && jobFinished(runningReport)) {
        reportJob = null;
        if (runningReport.status === "succeeded")
          report = await api(`/jobs/${initial.id}/training-report`);
        if (runningReport.status !== "succeeded")
          error.textContent =
            runningReport.error || "Evaluation was cancelled.";
      }
      if (!batch && !comparing && !report && job.status === "succeeded")
        report = await api(`/jobs/${initial.id}/training-report`);
      if (!alive()) return;
      host.querySelector("[data-run-status]").innerHTML =
        `${badge(reportJob ? "running" : job.status)}<p>${reportJob ? "Evaluating the saved model…" : esc(job.error || job.progress_message || "")}</p>${!jobFinished(job) ? `<progress value="${job.progress}" max="1" aria-label="${title} progress"></progress><small>${Math.round(job.progress * 100)}%</small>` : ""}`;
      if (logs.entries.length && limit === Number(count.value)) {
        cursor = logs.next_cursor;
        entries = [...entries, ...logs.entries].slice(-limit);
        output.value = entries
          .map(
            (e) =>
              `${new Date(e.created_at).toLocaleTimeString()} ${e.level === "error" ? "ERROR " : ""}${e.message}`,
          )
          .join("\n");
        if (host.querySelector("[data-follow-log]").checked)
          output.scrollTop = output.scrollHeight;
      }
      host.querySelector("[data-log-note]").textContent = logs.total_lines
        ? `Showing ${Math.min(entries.length, limit).toLocaleString()} of ${logs.total_lines.toLocaleString()} lines. Download includes the full log.`
        : "";
      if (!entries.length)
        output.value = jobFinished(job)
          ? "Detailed logs were not recorded for this earlier run."
          : `Waiting for ${title.toLowerCase()} logs…`;
      if (batch) {
        const result = job.result || {};
        host.querySelector("[data-report]").innerHTML =
          `<p>${result.asset_ids?.length || 0} images segmented · ${result.annotation_ids?.length || 0} pending reviews · ${result.failed?.length || 0} failed</p>`;
      }
      if (report && report.id !== reportId) {
        reportId = report.id;
        host.querySelector("[data-report]").innerHTML = (
          comparing ? comparisonHTML : resultHTML
        )(report, state);
        const button = host.querySelector("[data-download-report]");
        if (button)
          button.onclick = () =>
            download(
              JSON.stringify(report, null, 2),
              `evaluation-${initial.id}.json`,
              "application/json",
            );
      }
      run.hidden =
        !training ||
        !canManage() ||
        job.status !== "succeeded" ||
        report?.evaluation_requested === false ||
        Boolean(report?.metrics);
      run.disabled = Boolean(reportJob);
      run.textContent = reportJob
        ? "Evaluating…"
        : report
          ? "Retry evaluation"
          : "Evaluate model";
      host.querySelector("[data-report-note]").textContent =
        training && !report && job.status === "succeeded"
          ? "This earlier run has no evaluation report. Evaluate its saved model without retraining."
          : "";
      if (!jobFinished(job) || reportJob) timer = setTimeout(poll, 1000);
    } catch (failure) {
      if (alive()) error.textContent = failure.message;
    } finally {
      busy = false;
      if (refreshPending && alive()) {
        refreshPending = false;
        clearTimeout(timer);
        timer = setTimeout(poll, 0);
      }
    }
  }
  run.onclick = async () => {
    run.disabled = true;
    error.textContent = "";
    try {
      const job = await api(`/jobs/${initial.id}/training-report`, "POST", {});
      if (!alive()) return;
      reportJob = job.id;
      await poll();
    } catch (failure) {
      if (alive()) error.textContent = failure.message;
    } finally {
      if (alive()) run.disabled = Boolean(reportJob);
    }
  };
  void poll();
}
