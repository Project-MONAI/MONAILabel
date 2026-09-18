import { escapeHTML as esc, badge, button } from "./ui.js";
import { upload } from "./dataset-import.js";
import { reserveViewerTab, closePendingViewer } from "./viewer-launch.js";

export function videoButtons(state, asset) {
  if (!state.roles.some((role) => ["annotator", "manager"].includes(role)))
    return "";
  if (!state.videoCapabilities?.cvat)
    return '<span class="muted">CVAT unavailable</span>';
  return (
    button("CVAT", "video-open", asset.id) +
    button("Submit saved tracks", "video-submit", asset.id)
  );
}

export function videoReviews(state, statusOf) {
  const query = state.searches.review.trim().toLowerCase();
  const videos = (state.videos || []).filter(
    (v) =>
      v.annotation_id &&
      `${v.name} ${v.group_id}`.toLowerCase().includes(query) &&
      (state.reviewFilter === "all" || statusOf(v) === state.reviewFilter),
  );
  const reviewer = state.roles.some((role) =>
    ["reviewer", "manager"].includes(role),
  );
  return `<section class="video-list"><div class="section-heading"><h2>Video clips</h2></div><p class="muted">Review submitted tool tracks, or open CVAT to inspect and correct them.</p>${videos.length ? `<div class="table-wrap"><table class="compact-table"><thead><tr><th>Clip</th><th>Status</th><th>Actions</th></tr></thead><tbody>${videos.map((v) => `<tr><td>${esc(v.name)}<small>${v.width} × ${v.height} · ${v.frames} frames · ${v.duration.toFixed(2)} s · ${esc(v.group_id)} · Revision ${v.revision}</small></td><td>${badge(statusOf(v))} ${badge(v.split === "validation" ? "Evaluation only" : "Annotation")}</td><td><div class="row-actions">${state.videoCapabilities?.cvat && reviewer ? button("Inspect in CVAT", "video-inspect", v.id) + button("Submit saved tracks", "video-submit", v.id) : ""}${reviewer ? button("Review revision", "video-decision", v.id) : ""}${button("Track data", "video-tracks", v.id)}</div></td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">No video clips match this view.</p>'}</section>`;
}

export async function videoAction(name, id, ui) {
  const { state, api, modal, field, selectField, message, watch, safely } = ui;
  const projectId = state.project.id;
  if (name === "video-import") {
    return modal(
      "Import video",
      field(
        "Video clip",
        "clip",
        "file",
        "required accept='.mp4,.mov,.mkv,.avi,.webm'",
        "Up to 2 GiB; source frame timestamps are preserved.",
      ) +
        field(
          "Patient or procedure ID",
          "group_id",
          "text",
          "required maxlength='200'",
          "Use the same ID for related clips and exported images.",
        ) +
        field(
          "Instrument labels",
          "labels",
          "text",
          "maxlength='1000' placeholder='Grasper, Scissors'",
          "Comma-separated names; existing project labels are kept.",
        ) +
        selectField(
          "Dataset use",
          "split",
          '<option value="pool">Annotation</option><option value="validation">Evaluation only (reserved)</option>',
        ) +
        '<p data-video-progress role="status"></p>',
      async (form) => {
        const file = form.get("clip");
        const params = new URLSearchParams({
          name: file.name,
          group_id: form.get("group_id").trim(),
          split: form.get("split"),
        });
        for (const label of form
          .get("labels")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean))
          params.append("labels", label);
        await upload(
          `/projects/${projectId}/videos/upload?${params}`,
          file,
          (fraction) => {
            document.querySelector("[data-video-progress]").textContent =
              fraction === 1
                ? "Reading video frames…"
                : `Uploading ${Math.round(fraction * 100)}%`;
          },
        );
        message("Video imported. Open CVAT to annotate instrument tracks.");
      },
      "Import video",
    );
  }
  const asset = state.videos.find((v) => v.id === id);
  if (!asset) throw new Error("This video is no longer available.");
  if (name === "video-open" || name === "video-inspect") {
    const viewerTab = reserveViewerTab();
    try {
      const job = await api(`/videos/${id}/editor`, "POST", {
        base_revision: asset.revision,
        mode: name === "video-inspect" ? "review" : "annotation",
      });
      message(
        state.videoCapabilities?.managed
          ? "Preparing CVAT. The first launch downloads and starts its services. The editor will open automatically when ready."
          : "Preparing CVAT. The editor will open automatically when ready; sign in to CVAT if prompted.",
      );
      safely(() => watch(job.id, projectId, viewerTab));
    } catch (error) {
      closePendingViewer(viewerTab);
      throw error;
    }
    return;
  }
  if (name === "video-submit") {
    const editors = (await api(`/videos/${id}/editors`)).filter(
      (e) => e.base_revision === asset.revision && !e.submitted_annotation_id,
    );
    if (!editors.length)
      throw new Error(
        "Open CVAT for this revision first, then save your tracks there.",
      );
    return modal(
      "Submit saved CVAT tracks",
      `<p>${esc(asset.name)} · revision ${asset.revision}</p><p>Save in CVAT before continuing. Browser edits that have not been saved in CVAT cannot be submitted. This creates a new pending review revision.</p>` +
        selectField(
          "CVAT task",
          "editor",
          editors
            .map(
              (e) =>
                `<option value="${e.id}">${esc(e.mode)} · task ${e.task_id}</option>`,
            )
            .join(""),
        ),
      async (form) => {
        await api(`/videos/${id}/cvat-submit`, "POST", {
          editor_id: form.get("editor"),
        });
        message(
          "Saved tracks submitted for review. Existing CVAT tasks remain available with their drafts.",
        );
      },
      "Submit for review",
    );
  }
  if (name === "video-decision") {
    return modal(
      "Review submitted video revision",
      `<p>${esc(asset.name)} · revision ${asset.revision}</p><p>This decision applies to the submitted tracks. Submit any corrections saved in CVAT before accepting them.</p>` +
        selectField(
          "Decision",
          "verdict",
          '<option value="accepted">Good</option><option value="changes_requested">Needs changes</option>',
        ) +
        field("Comment (optional)", "comment"),
      async (form) => {
        await api(`/videos/${id}/decision`, "POST", {
          base_revision: asset.revision,
          verdict: form.get("verdict"),
          comment: form.get("comment") || "",
        });
      },
      "Save decision",
    );
  }
  if (name === "video-tracks") {
    const tracks = await api(`/videos/${id}/tracks`);
    return modal(
      "Submitted track data",
      `<p>${esc(asset.name)} · revision ${tracks.base_revision}</p><pre>${esc(JSON.stringify(tracks.document, null, 2))}</pre>`,
      async () => {},
      "Close",
    );
  }
}
