import { escapeHTML as esc } from "./ui.js";
import { reviewItems } from "./review-items.js";

export function inspectReview(
  id,
  { state, modal, api, refresh, latestDecision, openViewer },
) {
  const item = reviewItems(state).find((u) => u.unit_id === id);
  if (!item) throw new Error("This review item is no longer available.");
  const canReview = state.roles.includes("reviewer");
  const canAnnotate = state.roles.includes("annotator");
  const prior = latestDecision(item);
  const frames = item.scope.kind === "frames";
  const range = item.scope;
  const description = frames
    ? `Frames ${range.start}–${range.stop - 1}`
    : `Region at ${range.region.x}, ${range.region.y} · ${range.region.width} × ${range.region.height} pixels`;
  modal(
    item.name,
    `<section data-review-inspection><p>${esc(description)} · Revision ${item.revision}</p><img class="review-preview" alt="Submitted annotation over its source image"><p data-preview-status role="status"></p><label class="checkbox-row"><input type="checkbox" data-review-overlay checked>Show segmentation</label>${frames ? `<label>Source frame<input type="range" data-review-frame min="${range.start}" max="${range.stop - 1}" value="${range.start}" step="1"><output data-frame-number>${range.start}</output></label>` : ""}<button type="button" data-review-edit>Open ${frames ? "CVAT" : "QuPath"} to correct</button>${canReview ? `<label for="review-verdict">Decision</label><select id="review-verdict" name="verdict"><option value="pending">Pending review</option><option value="accepted">Good</option><option value="changes_requested">Needs changes</option></select><label>Comment (optional)<textarea name="comment" maxlength="4000">${esc(prior?.comment || "")}</textarea></label>` : ""}</section>`,
    async (form) => {
      if (canReview) {
        await api(`/review-units/${id}/decision`, "POST", {
          base_revision: item.revision,
          decision_id: prior?.id || null,
          verdict: form.get("verdict"),
          comment: form.get("comment") || "",
        });
        await refresh();
      }
    },
    canReview ? "Save review" : "Close",
  );
  const host = document.querySelector("[data-review-inspection]");
  const picture = host.querySelector("img");
  const status = host.querySelector("[data-preview-status]");
  const slider = host.querySelector("[data-review-frame]");
  const overlay = host.querySelector("[data-review-overlay]");
  const verdict = host.querySelector('[name="verdict"]');
  if (verdict) verdict.value = prior?.verdict || "pending";
  const load = () => {
    const params = new URLSearchParams({
      base_revision: item.revision,
      overlay: overlay.checked,
    });
    if (slider) {
      params.set("frame", slider.value);
      host.querySelector("[data-frame-number]").textContent = slider.value;
    }
    status.textContent = "Loading source pixels…";
    picture.onload = () => {
      status.textContent = "";
    };
    picture.onerror = () => {
      status.textContent =
        "Preview unavailable. Reopen this review item or inspect it in the viewer.";
    };
    picture.src = `/api/review-units/${id}/preview?${params}`;
  };
  overlay.onchange = load;
  if (slider) slider.onchange = load;
  const edit = host.querySelector("[data-review-edit]");
  edit.hidden = !canAnnotate;
  edit.onclick = () => {
    host.closest("dialog").close();
    openViewer(item);
  };
  load();
}
