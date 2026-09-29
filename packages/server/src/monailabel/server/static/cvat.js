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
    viewer_actions:
      canEdit() && !info.editor.submitted_annotation_id
        ? [
            "clear_video_annotations",
            "video_interaction",
            "undo",
            "redo",
            "save_draft",
            "submit",
          ]
        : [],
    video: {
      hint_revision: hintRevision,
      object_key: objectIdentity,
      video_id: info.video.id,
      editor_id: editorId,
      frame: native.frame,
      client_id: native.client_id,
      label_id: labelId ? Number(labelId) : null,
      available_label_ids: Object.entries(info.editor.label_map)
        .filter(([, id]) => native.labels.includes(id))
        .map(([id]) => Number(id)),
      box: native.box,
      points: native.points,
      occluded: native.occluded,
      draft_signature: native.draft_signature,
    },
    base_revision: info.editor.base_revision,
    model_id: $("#detection-model").value || null,
    interaction_mode: inputMode,
    interaction_target: $("#target-label").value.trim(),
    spatial_prompt: selectedModel()?.interaction?.video_scopes?.length
      ? currentSpatial()
      : null,
    label_ids: labelId
      ? [Number(labelId)]
      : labels.length === 1
        ? [labels[0].id]
        : [],
  };
}
async function watch(jobId, guard) {
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
          if (guard && guard !== interactionSignature())
            throw new Error(
              "The frame, target or inputs changed. Run Update again; your draft is preserved.",
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
    const addedId = await bridge().apply(
      proposal,
      info.editor.label_map[proposal.request.label_id],
    );
    if (addedId != null && proposal.request.spatial_prompt) {
      const old = objectIdentity;
      objectIdentity = String(addedId);
      labelObjects.set(
        $("#target-label").value.trim().toLocaleLowerCase(),
        objectIdentity,
      );
      for (const [key, hints] of [...inputHints]) {
        const parts = JSON.parse(key);
        if (parts[0] === old) {
          parts[0] = objectIdentity;
          inputHints.set(JSON.stringify(parts), hints);
          inputHints.delete(key);
        }
      }
      hintRevision = randomId();
      refreshObjects();
    }
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
async function editDraft(action) {
  if (writing)
    throw new Error("Wait for the current draft operation to finish.");
  writing = true;
  iframe.inert = true;
  try {
    info = await api(`/cvat/editors/${editorId}`);
    if (
      !canEdit() ||
      info.editor.submitted_annotation_id ||
      action.project_id !== info.project.id ||
      action.video_id !== info.video.id ||
      action.editor_id !== editorId ||
      action.base_revision !== info.video.revision ||
      action.base_revision !== info.editor.base_revision
    )
      throw new Error(
        "This video session changed. Reopen it; your draft is preserved.",
      );
    if ((await bridge().snapshot()) !== action.draft_signature)
      throw new Error(
        "Your CVAT draft changed. Retry the request to keep your edits.",
      );
    if (action.client_action === "clear_video_annotations") {
      const count = await bridge().clear(
        action,
        action.label_ids.map((id) => info.editor.label_map[id]),
      );
      message(
        count
          ? "Cleared the requested annotations. Say ‘undo’ to restore them."
          : "No matching annotations on the requested frames.",
      );
      status(count ? "Draft updated. Submit when ready." : "Draft unchanged.");
    } else if (["undo", "redo"].includes(action.operation)) {
      await bridge().history(action.operation, action.draft_signature);
      status(
        action.operation === "undo"
          ? "Undid the last edit."
          : "Redid the last edit.",
      );
    } else if (["save_draft", "submit"].includes(action.operation)) {
      // save() owns the same write lock and rechecks submission on the server.
      writing = false;
      await save(action.operation === "submit");
    } else throw new Error("Unsupported CVAT draft operation.");
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
async function refreshLabels() {
  writing = true;
  iframe.inert = true;
  try {
    const previous = bridge();
    const frame = previous.frame();
    // Save this task's draft before replacing the native session's label list.
    // This does not submit annotations or change their review status.
    await previous.save();
    info = await api(`/cvat/editors/${editorId}`);
    iframe.contentWindow.location.reload();
    const deadline = Date.now() + 60000;
    while (bridge() === previous || !bridge()?.ready()) {
      if (Date.now() > deadline)
        throw new Error(
          "The viewer is still loading. Your draft is saved; retry when ready.",
        );
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    await bridge().setFrame(frame);
  } finally {
    writing = false;
    iframe.inert = false;
  }
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
    for (let attempt = 0; attempt < 2; attempt++) {
      const reply = await api(
        `/projects/${info.project.id}/assistant`,
        "POST",
        {
          message: prompt,
          context: await currentContext(),
          conversation_id: conversation || null,
          request_id: randomId(),
        },
      );
      conversation = reply.conversation_id;
      message(reply.message);
      if (reply.data?.client_action === "refresh_video_labels") {
        status("Updating viewer labels…");
        await refreshLabels();
        if (attempt)
          throw new Error(
            "Viewer labels updated. Please repeat your annotation prompt.",
          );
        continue;
      }
      if (reply.data?.client_action === "video_interaction")
        await applyInputAction(reply.data);
      if (
        ["clear_video_annotations", "video_edit"].includes(
          reply.data?.client_action,
        )
      )
        await editDraft(reply.data);
      if (reply.job_id) await watch(reply.job_id);
      break;
    }
  } finally {
    chatting = false;
    $("#send").disabled = false;
  }
});
let hintRevision = randomId();
let inputMode = "navigate",
  objectIdentity = "new-" + randomId();
const inputHints = new Map(),
  labelObjects = new Map();
let hintRender = "",
  objectsRender = "";
const selectedModel = () =>
  info?.detection_models.find((m) => m.id === $("#detection-model").value);
const hintKey = () =>
  JSON.stringify([
    objectIdentity,
    $("#target-label").value.trim().toLocaleLowerCase(),
    bridge().frame(),
  ]);
function currentSpatial() {
  const hints = inputHints.get(hintKey()) || [];
  const points = hints
    .filter((h) => h.kind === "point")
    .map((h) => ({
      coordinates: [...h.coordinates[0]].reverse(),
      positive: h.positive,
    }));
  const box =
    hints
      .find((h) => h.kind === "box")
      ?.coordinates.map((p) => [...p].reverse()) || null;
  return box || points.some((p) => p.positive) ? { points, box } : null;
}
const interactionSignature = () =>
  JSON.stringify([
    hintKey(),
    selectedModel()?.id,
    inputHints.get(hintKey()) || [],
  ]);
function refreshObjects() {
  const label = info.project.labels.find(
    (l) =>
      l.name.toLocaleLowerCase() ===
      $("#target-label").value.trim().toLocaleLowerCase(),
  );
  const objects = (bridge()?.objects() || []).filter(
    (o) => o.label === info.editor.label_map[label?.id],
  );
  const signature = JSON.stringify([objects, objectIdentity]);
  if (signature === objectsRender) return;
  objectsRender = signature;
  $("#target-object").replaceChildren(
    new Option("New object", "new"),
    ...objects.map(
      (o) => new Option(`${o.name} · object ${o.id}`, String(o.id)),
    ),
  );
  $("#target-object").value = objects.some(
    (o) => String(o.id) === objectIdentity,
  )
    ? objectIdentity
    : "new";
}
function configureModel() {
  inputMode = "navigate";
  hintRender = "";
  const model = selectedModel();
  const inputs = model?.interaction?.video_scopes?.length
    ? model.interaction.inputs
    : {};
  $("#input-tools").replaceChildren();
  const paths = {
    positive: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M8 12h8 M12 8v8",
    negative: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M8 12h8",
    box: "M5 3H3v2 M8 3h8 M19 3h2v2 M21 8v8 M21 19v2h-2 M16 21H8 M5 21H3v-2 M3 16V8",
  };
  for (const [mode, kind, label] of [
    ["positive", "positive_point", "+ Point"],
    ["negative", "negative_point", "− Point"],
    ["box", "box", "Box"],
  ]) {
    if (!inputs[kind]) continue;
    const button = document.createElement("button");
    button.dataset.mode = mode;
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", "false");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    const path = document.createElementNS(svg.namespaceURI, "path");
    path.setAttribute("d", paths[mode]);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.8");
    svg.append(path);
    button.append(svg);
    button.onclick = () => {
      inputMode = inputMode === mode ? "navigate" : mode;
      syncInteraction();
    };
    $("#input-tools").append(button);
  }
  $("#input-separator").hidden = !$("#input-tools").children.length;
  const names =
    model?.supported_targets ??
    info.project.labels.filter((l) => l.id).map((l) => l.name);
  $("#target-labels").replaceChildren(
    ...names.map((name) => new Option(name, name)),
  );
  $("#target-object").hidden = !Object.keys(inputs).length;
}
function syncInteraction() {
  refreshObjects();
  const disabled =
    chatting ||
    writing ||
    !!activeJob ||
    !canEdit() ||
    !!info.editor.submitted_annotation_id;
  for (const id of [
    "detection-model",
    "target-label",
    "target-object",
    "track-count",
  ])
    $("#" + id).disabled = disabled;
  const hasTarget = !!$("#target-label").value.trim();
  for (const button of $("#input-tools").children) {
    button.disabled = disabled || !hasTarget;
    button.setAttribute(
      "aria-pressed",
      String(inputMode === button.dataset.mode),
    );
  }
  $("#update-frame").disabled = disabled || !hasTarget;
  $("#track-range").disabled = disabled || !hasTarget;
  $("#update-options summary").setAttribute("aria-disabled", String(disabled));
  const interactive = !!selectedModel()?.interaction?.video_scopes?.length;
  const options = {
    key: hintKey(),
    mode: interactive ? inputMode : "navigate",
    hints: interactive ? inputHints.get(hintKey()) || [] : [],
    width: info.video.width,
    height: info.video.height,
    disabled: disabled || !interactive,
    geometry: bridge().frame(),
  };
  const signature = JSON.stringify(options);
  if (signature !== hintRender) {
    hintRender = signature;
    const key = options.key;
    bridge()
      .configureHints(
        options,
        (hints) => {
          inputHints.set(key, hints);
          hintRevision = randomId();
          hintRender = "";
        },
        () => {
          inputMode = "navigate";
          hintRender = "";
        },
      )
      .catch((error) => status(error.message, true));
  }
}
$("#detection-model").onchange = configureModel;
$("#target-label").oninput = () => {
  inputMode = "navigate";
  const label = $("#target-label").value.trim().toLocaleLowerCase();
  if (!labelObjects.has(label)) labelObjects.set(label, "new-" + randomId());
  objectIdentity = labelObjects.get(label);
  hintRender = "";
};
$("#target-object").onchange = () => {
  objectIdentity =
    $("#target-object").value === "new"
      ? "new-" + randomId()
      : $("#target-object").value;
  labelObjects.set(
    $("#target-label").value.trim().toLocaleLowerCase(),
    objectIdentity,
  );
  inputMode = "navigate";
  hintRender = "";
};
$("#update-options summary").onclick = (event) => {
  if (event.currentTarget.getAttribute("aria-disabled") === "true")
    event.preventDefault();
};
document.addEventListener("click", (event) => {
  if (!$("#update-options").contains(event.target))
    $("#update-options").open = false;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    inputMode = "navigate";
    $("#update-options").open = false;
  }
});
async function applyInputAction(action) {
  if (
    action.video_id !== info.video.id ||
    action.editor_id !== editorId ||
    action.base_revision !== info.editor.base_revision ||
    action.frame !== bridge().frame() ||
    action.hint_revision !== hintRevision ||
    action.object_key !== objectIdentity ||
    action.draft_signature !== (await bridge().snapshot())
  )
    throw new Error(
      "The viewer changed. Retry the input request; your work is preserved.",
    );
  if (action.operation === "mode") {
    if (action.model_id) {
      $("#detection-model").value = action.model_id;
      configureModel();
    }
    if (action.target) {
      $("#target-label").value = action.target;
      $("#target-label").oninput();
    }
    inputMode = action.mode;
  } else {
    for (const [key, hints] of inputHints) {
      const [object, , frame] = JSON.parse(key);
      if (
        (!action.all_objects && object !== objectIdentity) ||
        (action.scope === "current_frame" && frame !== action.frame)
      )
        continue;
      inputHints.set(
        key,
        hints.filter(
          (h) =>
            !(
              (action.kind === "all" || action.kind === h.kind) &&
              (action.polarity === "all" ||
                (h.kind === "point" &&
                  h.positive === (action.polarity === "positive")))
            ),
        ),
      );
    }
    hintRevision = randomId();
  }
  hintRender = "";
  syncInteraction();
}

async function updateVideo(track) {
  if (chatting || writing || activeJob) return;
  const model = selectedModel(),
    target = $("#target-label").value.trim();
  if (!model || !target)
    throw new Error("Choose a model and target label first.");
  const interactive = !!model.interaction?.video_scopes?.length;
  const hints = inputHints.get(hintKey()) || [];
  const points = hints
    .filter((h) => h.kind === "point")
    .map((h) => ({
      coordinates: [...h.coordinates[0]].reverse(),
      positive: h.positive,
    }));
  const box = hints
    .find((h) => h.kind === "box")
    ?.coordinates.map((p) => [...p].reverse());
  if (interactive && !box && !points.some((p) => p.positive))
    throw new Error("Add a positive point or draw a box around one object.");
  const frame = bridge().frame();
  const count = track ? Number($("#track-count").value) : 1;
  if (
    !Number.isInteger(count) ||
    count < 1 ||
    frame + count > info.video.frames
  )
    throw new Error("Choose a tracking range within the clip.");
  chatting = true;
  inputMode = "navigate";
  $("#update-options").open = false;
  status(track ? "Preparing tracking…" : "Preparing segmentation…");
  syncInteraction();
  try {
    const result = await api(
      `/videos/${info.video.id}/editors/${editorId}/target`,
      "POST",
      { base_revision: info.editor.base_revision, model_id: model.id, target },
    );
    if (result.refresh) await refreshLabels();
    const native = await bridge().context();
    if (native.frame !== frame)
      throw new Error("The frame changed. Retry Update on the intended frame.");
    const request = {
      editor_id: editorId,
      base_revision: info.editor.base_revision,
      model_id: model.id,
      label_id: result.label_id,
      frame,
      frame_count: count,
      draft_signature: native.draft_signature,
    };
    if (interactive) {
      request.spatial_prompt = { points, box: box || null };
      request.client_id = objectIdentity.startsWith("new-")
        ? null
        : Number(objectIdentity);
    } else {
      request.output = "polygon";
      request.prompt = "Segment one visible instance of " + target;
    }
    const guard = interactionSignature();
    const job = await api(
      `/videos/${info.video.id}/${interactive ? "interactive" : "find-and-track"}`,
      "POST",
      request,
    );
    await watch(job.id, guard);
  } finally {
    chatting = false;
    hintRender = "";
  }
}
$("#update-frame").onclick = safely(() => updateVideo(false));
$("#track-range").onclick = safely(() => updateVideo(true));

async function start() {
  info = await api(`/cvat/editors/${editorId}`);
  $("#title").textContent = info.video.name;
  $("#revision").textContent =
    `Revision ${info.editor.base_revision} · ${info.editor.mode}`;
  $("#back").href = `/datasets?project=${info.project.id}`;
  const groups = [
    ...new Set(
      info.detection_models.map((m) => m.catalog_group || "Other models"),
    ),
  ];
  for (const [index, group] of groups.entries()) {
    if (index) $("#detection-model").append(document.createElement("hr"));
    const section = document.createElement("optgroup");
    section.label = group;
    for (const model of info.detection_models.filter(
      (m) => (m.catalog_group || "Other models") === group,
    ))
      section.append(new Option(model.name, model.id));
    $("#detection-model").append(section);
  }
  $("#target-label").value =
    info.project.labels.filter((l) => l.id).length === 1
      ? info.project.labels.find((l) => l.id).name
      : "";
  if (
    info.detection_models.some((m) => m.id === info.project.annotation_model_id)
  )
    $("#detection-model").value = info.project.annotation_model_id;
  labelObjects.set(
    $("#target-label").value.trim().toLocaleLowerCase(),
    objectIdentity,
  );
  configureModel();
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
      syncInteraction();
      $("#selection").textContent =
        `Frame ${bridge().frame()} of ${info.video.frames - 1}` +
        (bridge().hasSelection()
          ? ` · ${bridge().selection()}`
          : " · No tool selected");
    }
  }, 250);
}
safely(start)();
