const decisions = {
  accepted: "Good",
  changes_requested: "Needs changes",
  pending: "Pending",
};

export function reviewDecisionControl() {
  return `<select class="review-decision" data-review-decision aria-label="Review decision for selected reviews"><option value="" selected disabled>Review decision…</option>${Object.entries(
    decisions,
  )
    .map(([value, text]) => `<option value="${value}">${text}</option>`)
    .join("")}</select>`;
}

// Decisions apply to saved immutable annotations and observed decision revisions.
export async function reviewAction(verdict, ui) {
  const { state, latestDecision, api, refresh, notice, modal } = ui;
  if (!Object.hasOwn(decisions, verdict))
    throw new Error("Choose a review decision.");
  if (!state.roles.includes("reviewer"))
    throw new Error("Reviewer access is required.");
  const assets = state.assets.filter(
    (a) => a.annotation_id && state.selectedReviews.has(a.annotation_id),
  );
  if (!assets.length) throw new Error("Select saved annotations to review.");
  const items = assets.map((a) => ({
    annotation_id: a.annotation_id,
    decision_id: latestDecision(a)?.id || null,
  }));
  const label = decisions[verdict];
  const apply = async (comment) => {
    const result = await api(
      `/projects/${state.project.id}/review-decisions`,
      "POST",
      { items, verdict, comment },
    );
    state.selectedReviews.clear();
    await refresh();
    notice(
      `${result.length} review${result.length === 1 ? "" : "s"} marked ${label}.`,
    );
  };
  if (verdict === "changes_requested")
    return modal(
      `Mark ${assets.length} review${assets.length === 1 ? "" : "s"} as Needs changes`,
      '<label>Review comment (optional)<textarea name="comment" maxlength="4000" rows="3"></textarea></label>',
      (form) => apply(form.get("comment") || ""),
      "Mark needs changes",
    );
  await apply("");
}
