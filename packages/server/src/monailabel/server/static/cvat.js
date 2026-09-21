import { randomId } from "./random-id.js";

const $ = (s) => document.querySelector(s);
const editorId = location.pathname.split("/").at(-1);
let info,
  conversation,
  activeJob,
  chatting = false,
  writing = false;
const iframe = $("#editor");
const bridge = () => iframe.contentWindow?.monaiVideo;
const status = (message, error = false) => {
  $("#status").textContent = message;
  $("#status").classList.toggle("error", error);
};
$("#prompt").onkeydown = (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("#send").disabled) $("#chat").requestSubmit();
  }
};
async function api(path, method = "GET", body) {
  const response = await fetch("/api" + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "The request could not be completed.",
    );
  return result;
}
function safely(fn) {
  return async (event) => {
    event?.preventDefault();
    try {
      await fn();
    } catch (error) {
      status(error.message, true);
    }
  };
}
function canEdit() {
  return (
    info.roles.includes("manager") ||
    info.roles.includes(
      info.editor.mode === "review" ? "reviewer" : "annotator",
    )
  );
}
async function currentContext() {
  const native = await bridge().context();
  const labelId = Object.entries(info.editor.label_map).find(
    ([, cvatId]) => cvatId === native.label,
  )?.[0];
  const labels = info.project.labels.filter(
    (label) => label.id && info.editor.label_map[label.id],
  );
  return {
    video: {
      video_id: info.video.id,
      editor_id: editorId,
      frame: native.frame,
      client_id: native.client_id,
      label_id: labelId ? Number(labelId) : null,
      box: native.box,
      points: native.points,
      occluded: native.occluded,
      draft_signature: native.draft_signature,
    },
    base_revision: info.editor.base_revision,
    model_id: $("#detection-model").value || null,
    label_ids: labelId
      ? [Number(labelId)]
      : labels.length === 1
        ? [labels[0].id]
        : [],
  };
}
async function watch(jobId) {
  activeJob = jobId;
  $("#cancel").hidden = false;
  try {
    for (;;) {
      const job = await api(`/jobs/${jobId}`);
      status(job.progress_message || "Working…");
      if (job.status === "succeeded") {
        if (job.result.video_proposal_id) {
          const proposal = await api(
            `/videos/${info.video.id}/tracking-proposals/${job.result.video_proposal_id}`,
          );
          await applyResult(proposal);
        } else status("Done.");
        return;
      }
      if (["failed", "cancelled", "interrupted"].includes(job.status))
        throw new Error(job.error || `Job ${job.status}.`);
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
  } finally {
    activeJob = null;
    $("#cancel").hidden = true;
  }
}
$("#cancel").onclick = safely(async () => {
  if (activeJob) {
    await api(`/jobs/${activeJob}/cancel`, "POST", {});
    status("Cancellation requested.");
  }
});
async function applyResult(proposal) {
  if (writing)
    throw new Error("Wait for the current draft operation to finish.");
  writing = true;
  iframe.inert = true;
  try {
    // Revalidate the saved revision and native draft immediately before applying.
    await api(`/videos/${info.video.id}/tracking-proposals/${proposal.id}`);
    await bridge().apply(
      proposal,
      info.editor.label_map[proposal.request.label_id],
    );
    const keys = proposal.keyframes;
    const range =
      keys.length === 1
        ? `frame ${keys[0].frame}`
        : `frames ${keys[0].frame}–${keys.at(-1).frame}`;
    const shape =
      proposal.request.output === "polygon" ? "Segmentation" : "Bounding box";
    const source = proposal.detection
      ? ` using ${proposal.detection.model_name}`
      : "";
    const action = proposal.request.client_id === null ? "added" : "updated";
    message(
      `${shape} ${action} in your draft on ${range}${source}. Review and correct it before submitting.`,
    );
    const warnings = (proposal.warnings || []).join(" ");
    if (warnings) message(warnings);
    if (proposal.masks_key) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Segmentation details";
      const link = document.createElement("a");
      link.textContent = "Download original masks";
      link.href = `/api/videos/${info.video.id}/tracking-proposals/${proposal.id}/masks`;
      details.append(summary, link);
      $("#messages").append(details);
    }
    status("Annotation added to draft. Use CVAT Undo to revert.");
  } finally {
    writing = false;
    iframe.inert = false;
  }
}
async function save(submit) {
  if (writing) return;
  writing = true;
  iframe.inert = true;
  try {
    await bridge().save();
    if (submit) {
      const result = await api(`/videos/${info.video.id}/cvat-submit`, "POST", {
        editor_id: editorId,
      });
      info = await api(`/cvat/editors/${editorId}`);
      $("#revision").textContent = `Revision ${result.revision} · submitted`;
      $("#review").hidden = !info.roles.some((r) =>
        ["manager", "reviewer"].includes(r),
      );
      status(
        "Tracks submitted for review. Return to the workspace to open the next revision.",
      );
    } else status("CVAT draft saved.");
  } finally {
    writing = false;
    iframe.inert = false;
  }
}
$("#save").onclick = safely(() => save(false));
$("#submit").onclick = safely(() => save(true));
for (const [id, verdict] of [
  ["accept", "accepted"],
  ["changes", "changes_requested"],
])
  $("#" + id).onclick = safely(async () => {
    await api(`/videos/${info.video.id}/decision`, "POST", {
      base_revision: info.video.revision,
      verdict,
      comment: "",
    });
    status(
      verdict === "accepted"
        ? "Submitted revision accepted."
        : "Changes requested for the submitted revision.",
    );
  });
function message(text, user = false) {
  $("#welcome")?.remove();
  const node = document.createElement("p");
  node.textContent = text;
  node.className = user ? "user" : "assistant";
  $("#messages").append(node);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}
$("#chat").onsubmit = safely(async () => {
  if (chatting || writing) return;
  if (!bridge())
    throw new Error("Wait for the video to load before asking the assistant.");
  const prompt = $("#prompt").value.trim();
  if (!prompt) return;
  $("#send").disabled = true;
  chatting = true;
  message(prompt, true);
  $("#prompt").value = "";
  try {
    const reply = await api(`/projects/${info.project.id}/assistant`, "POST", {
      message: prompt,
      context: await currentContext(),
      conversation_id: conversation || null,
      request_id: randomId(),
    });
    conversation = reply.conversation_id;
    message(reply.message);
    if (reply.job_id) await watch(reply.job_id);
  } finally {
    chatting = false;
    $("#send").disabled = false;
  }
});
async function start() {
  info = await api(`/cvat/editors/${editorId}`);
  $("#title").textContent = info.video.name;
  $("#revision").textContent =
    `Revision ${info.editor.base_revision} · ${info.editor.mode}`;
  $("#back").href = `/datasets?project=${info.project.id}`;
  for (const model of info.detection_models)
    $("#detection-model").add(new Option(model.name, model.id));
  if (
    info.detection_models.some((m) => m.id === info.project.annotation_model_id)
  )
    $("#detection-model").value = info.project.annotation_model_id;
  $("#review").hidden = !(
    info.editor.mode === "review" &&
    info.video.annotation_id &&
    info.video.revision === info.editor.base_revision &&
    info.roles.some((r) => ["manager", "reviewer"].includes(r))
  );
  $("#review").open = info.editor.mode === "review";
  iframe.src = `/tasks/${info.editor.task_id}/jobs/${info.editor.job_id}`;
  setInterval(() => {
    const ready = Boolean(bridge()?.ready());
    const stale = Boolean(info.editor.submitted_annotation_id);
    $("#save").disabled = !ready || !canEdit() || chatting || writing;
    $("#submit").disabled =
      !ready || !canEdit() || chatting || writing || stale;
    $("#send").disabled = !ready || chatting || writing;
    if (ready) {
      $("#selection").textContent =
        `Frame ${bridge().frame()} of ${info.video.frames - 1}` +
        (bridge().hasSelection()
          ? ` · ${bridge().selection()}`
          : " · No tool selected");
    }
  }, 250);
}
safely(start)();
