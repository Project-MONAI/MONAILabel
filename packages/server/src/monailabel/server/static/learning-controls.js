// Button-driven learning uses the same API as assistant tools, without another model call.
export async function learningAction(name, id, ui) {
  const {
    state,
    api,
    modal,
    escapeHTML: esc,
    message,
    render,
    watch,
    safely,
  } = ui;
  const prefix = `/projects/${state.project.id}`;
  if (name === "snapshot") {
    const snapshot = await api(`${prefix}/snapshots`, "POST", {});
    state.context.snapshot_id = snapshot.id;
    message(
      `Saved a dataset snapshot with ${snapshot.samples.length} accepted samples.`,
    );
    render();
  } else if (name === "evaluate") {
    const { evaluation_version_id, model_id, baseline_id } = state.context;
    const active = state.evaluationSets.filter((s) => !s.archived);
    const selected =
      active.find((s) => s.id === state.context.evaluation_set_id) ||
      (active.length === 1 ? active[0] : null);
    if (!model_id || !baseline_id) throw new Error("Choose both models first.");
    if (active.length && !selected && !evaluation_version_id)
      throw new Error("Choose an evaluation set.");
    if (model_id === baseline_id)
      throw new Error("Choose different models to compare.");
    const projectId = state.project.id;
    const job = await api(`${prefix}/evaluate`, "POST", {
      evaluation_set_id: evaluation_version_id ? null : selected?.id || null,
      evaluation_version_id: evaluation_version_id || null,
      candidate_id: model_id,
      baseline_id,
    });
    state.page = "activity";
    safely(() => watch(job.id, projectId));
    await ui.refresh();
  } else if (name === "promote") {
    const evaluation = state.evaluations.find(
      (e) => e.id === (id || state.context.evaluation_id),
    );
    if (!evaluation?.eligible_labels.length)
      throw new Error(
        "Choose an evaluation with eligible targets before promoting a model.",
      );
    const model = state.models.find((m) => m.id === evaluation.candidate_id);
    const version = state.project.version;
    modal(
      "Change annotation defaults",
      `<p>Use <strong>${esc(model.name)}</strong> by default for these targets?</p><div class="checkboxes">${state.project.labels
        .filter((l) => evaluation.eligible_labels.includes(l.id))
        .map(
          (l) =>
            `<label><input type="checkbox" name="labels" value="${l.id}" checked>${esc(l.name)}</label>`,
        )
        .join("")}</div>`,
      async (form) => {
        const label_ids = form.getAll("labels").map(Number);
        if (!label_ids.length) throw new Error("Choose at least one target.");
        await api(`${prefix}/promote`, "POST", {
          evaluation_id: evaluation.id,
          label_ids,
          base_version: version,
        });
        message(
          `Annotation defaults updated to ${model.name} for the selected targets.`,
        );
      },
      "Change defaults",
    );
  }
}
