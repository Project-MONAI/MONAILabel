import { escapeHTML as esc } from "./ui.js";

export function trainingEvaluation(form, { state, learner }) {
  const record = state.modelSplits.find((s) => s.learner_id === learner.id);
  const sets = state.evaluationSets.filter(
    (s) => !s.archived || s.id === learner.evaluation_set_id,
  );
  const selected = sets.find((s) => s.id === learner.evaluation_set_id);
  const host = form.querySelector("#training-evaluation");
  const savedPercentage =
    record?.validation_mode !== "fixed" && record?.validation_groups.length;
  host.innerHTML = `<fieldset><legend>Evaluation</legend><label>Evaluation set<select name="training_evaluation" required><option value="percentage" ${selected ? "" : "selected"}>${savedPercentage ? "This model’s percentage-based set" : "Create percentage-based set"}</option>${sets.map((s) => `<option value="${s.id}" ${selected?.id === s.id ? "selected" : ""}>${esc(s.name)} · Fixed${s.archived ? " (archived)" : ""}</option>`).join("")}</select></label><div data-evaluation-percentage><label>Evaluation (%)<input name="validation_percentage" type="number" min="1" max="50" value="${record?.validation_percentage || 20}"></label></div><small data-evaluation-hint></small><details><summary>How evaluation sets work</summary><p>Fixed sets use the same reference images. Percentage-based sets belong to this model and grow as more annotations are accepted. Existing assignments stay fixed; related patient images stay together, so percentages are approximate. Each run saves its exact images and labels.</p></details></fieldset>`;
  const choice = form.elements.training_evaluation;
  const input = form.elements.validation_percentage;
  const update = () => {
    const percentage = choice.value === "percentage";
    host.querySelector("[data-evaluation-percentage]").hidden = !percentage;
    input.disabled = !percentage;
    input.required = percentage;
    host.querySelector("[data-evaluation-hint]").textContent = percentage
      ? "Adds newly reviewed images to maintain this ratio."
      : sets.find((s) => s.id === choice.value)?.archived
        ? "Restore this set or choose another."
        : "Uses the same evaluation images each run.";
  };
  choice.onchange = update;
  update();
}

export function trainingEvaluationRequest(form) {
  const selected = form.get("training_evaluation");
  if (selected === "percentage") {
    const percentage = Number(form.get("validation_percentage"));
    if (!Number.isInteger(percentage) || percentage < 1 || percentage > 50)
      throw new Error("Choose 1–50% for evaluation.");
    return { validation_percentage: percentage };
  }
  if (!selected) throw new Error("Choose an evaluation set.");
  return { evaluation_set_id: selected };
}

export function showModelSplit({ state, modal }, learnerId) {
  const learner = state.learners.find((l) => l.id === learnerId);
  const record = state.modelSplits.find((s) => s.learner_id === learnerId);
  if (!learner || !record) return;
  const rows = (groups, images) =>
    state.assets.filter(
      (a) =>
        a.split !== "validation" &&
        (groups.includes(a.group_id) || images.includes(a.image_key)),
    );
  const training = rows(record.training_groups, record.training_image_keys);
  const fixed = state.evaluationSets.find(
    (s) => s.id === learner.evaluation_set_id,
  );
  const validation = fixed
    ? state.assets.filter((a) => fixed.member_groups.includes(a.group_id))
    : rows(record.validation_groups, record.validation_image_keys);
  const list = (name, assets) =>
    `<details ${name === "Evaluation" ? "open" : ""}><summary>${name} · ${assets.length} images</summary><ul>${assets.map((a) => `<li>${esc(a.name)} <small>${esc(a.group_id)}</small></li>`).join("")}</ul></details>`;
  modal(
    `${learner.name} · Evaluation set`,
    `<p>${fixed ? `${esc(fixed.name)} · Fixed` : `Percentage-based · ${record.validation_percentage}% · Only for this model`}</p>${list("Evaluation", validation)}${list("Training", training)}`,
    async () => true,
    "Close",
  );
}
