export function modelAction(kind, id, ui) {
  if (
    ![
      "rename-learner",
      "delete-learner",
      "rename-model",
      "delete-model",
    ].includes(kind)
  )
    return false;
  const { state, modal, field, api, escapeHTML: esc, message } = ui;
  const setup = kind.endsWith("learner"),
    deleting = kind.startsWith("delete");
  const record = (setup ? state.learners : state.models).find(
    (m) => m.id === id,
  );
  const learnerId = setup ? record.id : record.learner_id;
  const learner = state.learners.find((item) => item.id === learnerId);
  const versions = learnerId
    ? state.models.filter((item) => item.learner_id === learnerId)
    : [];
  const related = [...versions, ...(learner ? [learner] : [])];
  const title = deleting ? "Delete model" : "Rename model";
  modal(
    title,
    `<p><strong>${esc(record.name)}</strong></p>` +
      (deleting
        ? `<p>${learnerId ? `Removes ${learner ? "the training setup" : "this model"} and ${versions.length} trained version${versions.length === 1 ? "" : "s"} from Annotation and Training.` : "Removes this model from the available choices."} Saved annotations, completed runs and history are retained.</p>` +
          field(
            "Type the model name to confirm",
            "confirmation_name",
            "text",
            "required autocomplete='off'",
          )
        : field(
            "Model name",
            "name",
            "text",
            `required maxlength="120" value="${esc(record.name)}"`,
          ) +
          "<small>Use this name when referring to the model in chat.</small>") +
      '<button type="button" id="cancel-model-action">Cancel</button>',
    async (form) => {
      const name = form.get("name")?.trim();
      if (!deleting && !name) throw new Error("Give the model a name.");
      await api(
        `/projects/${state.project.id}/${setup ? "learners" : "models"}/${id}`,
        deleting ? "DELETE" : "PATCH",
        {
          base_version: record.version,
          ...(deleting
            ? {
                confirmation_name: form.get("confirmation_name"),
                scope: "model",
                related_versions: learnerId
                  ? Object.fromEntries(
                      related.map((item) => [item.id, item.version]),
                    )
                  : null,
              }
            : { name }),
        },
      );
      if (deleting) {
        const removed = new Set([
          id,
          learnerId,
          ...versions.map((item) => item.id),
        ]);
        for (const key of ["learner_id", "model_id", "baseline_id"])
          if (removed.has(state.context[key])) delete state.context[key];
      }
      message(
        deleting
          ? `Deleted ${record.name}${learnerId ? " from Annotation and Training" : ""}.`
          : `Model renamed to ${name}.`,
      );
    },
    title,
  );
  document.querySelector("#cancel-model-action").onclick = () =>
    document.querySelector("#dialog").close();
  if (deleting) {
    const form = document.querySelector("#action-form"),
      submit = form.querySelector('[type="submit"]');
    submit.className = "danger";
    submit.disabled = true;
    form.elements.confirmation_name.oninput = (e) => {
      submit.disabled = e.target.value !== record.name;
    };
  }
  return true;
}
