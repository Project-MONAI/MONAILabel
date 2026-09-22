import {
  trainingSamples,
  bindTrainingSamples,
  trainingSampleRequest,
} from "./training-samples.js";
import { importFiles } from "./dataset-import.js";
import { randomId } from "./random-id.js";
import { videoAction } from "./videos.js";
import {
  reserveViewerTab,
  openPreparedViewer,
  closePendingViewer,
  desktopTarget,
} from "./viewer-launch.js";
import { trainingResults } from "./training-results.js";
("use strict");
import { actionLabel, submitLabel, staticIcons } from "./icons.js";
import { evaluationSetAction } from "./evaluation-sets.js";
import {
  trainingEvaluation,
  trainingEvaluationRequest,
  showModelSplit,
} from "./training-evaluation.js";
import { reviewAction } from "./review-controls.js";
import { reviewItems, reviewStatus } from "./review-items.js";
import { inspectReview } from "./review-inspection.js";
import { learningAction } from "./learning-controls.js";
import { voiceInput } from "./voice-input.js";
import {
  trainingSettings,
  trainingOverrides,
  bindTrainingSettings,
} from "./training-settings.js";
import { modelAction } from "./model-actions.js";
import { setupModel } from "./model-setup.js";
import { importHostedModel } from "./model-import.js";
import { deleteProject, deleteFiles } from "./deletion.js";
import { escapeHTML, badge, button, jobName } from "./ui.js";
import {
  datasets,
  samples,
  sampleType,
  sampleDimensions,
  sampleUse,
  reviewQueue,
  activity,
  team,
  visibleAssets,
  visibleReviews,
  pageSlice,
} from "./workspace-lists.js";
import { modelLibrary } from "./model-library.js";
import { showModelTargets } from "./model-targets.js";
import { structurePicker } from "./structure-picker.js";
import { importDatasetTemplate } from "./dataset-templates.js";
import { openDicomImport } from "./dicom-import.js";
import { pageFromURL, pageURL } from "./navigation.js";
// A small server-backed console: no build step, UI framework, or client-side secrets.
const $ = (selector) => document.querySelector(selector);
const desktopLaunchTarget = desktopTarget();
const workspaceId = randomId();
let viewerReturns = null;
try {
  if ("BroadcastChannel" in window)
    viewerReturns = new BroadcastChannel("monailabel-workspace-" + workspaceId);
} catch {
  // The viewer can return in its own tab when browser storage is restricted.
}
const voice = voiceInput(
  $("#chat-input"),
  $("#dictate"),
  $("#voice-status"),
  $("#read-replies"),
);
const state = {
  user: null,
  projects: [],
  project: null,
  assets: [],
  videos: [],
  videoCapabilities: {},
  dicomSeries: [],
  models: [],
  learners: [],
  recipes: [],
  jobs: [],
  decisions: [],
  reviewUnits: [],
  evaluations: [],
  evaluationSets: [],
  evaluationVersions: [],
  selectedReviews: new Set(),
  credentials: [],
  roles: [],
  page: pageFromURL(new URL(location.href)),
  context: {},
  setup: false,
  conversations: new Map(),
  chatBusy: false,
  reviewFilter: "pending",
  selectedFiles: new Set(),
  members: [],
  pages: { datasets: 1, review: 1, activity: 1 },
  searches: { datasets: "", review: "", activity: "" },
  datasetFilter: "all",
  activityFilter: "all",
  modelsTab: "annotation",
};
const titles = {
  overview: "Overview",
  datasets: "Datasets",
  models: "Models",
  review: "Reviews",
  activity: "Activity",
  team: "Team & roles",
};
let replaceNavigation = true;
const canManage = () => state.roles.includes("manager");
const selectedAsset = () =>
  state.assets.find((a) => a.id === state.context.asset_id);
const latestDecision = (a) =>
  state.decisions.filter((d) => d.annotation_id === a.annotation_id).at(-1);
const statusOf = (a) => reviewStatus(state, a, latestDecision);

async function api(path, method = "GET", body) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error(
      "Cannot reach the server. Check your connection and retry.",
    );
  }
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401) showAuth();
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail),
    );
  }
  return result;
}

function notice(message) {
  $("#notice").textContent = message;
  $("#notice").classList.toggle("hidden", !message);
}
function message(text, role = "assistant", action = null) {
  const div = document.createElement("div");
  div.className = `message ${role}-message`;
  const p = document.createElement("p");
  p.textContent = text;
  div.append(p);
  if (action) {
    const b = document.createElement("button");
    b.textContent = action.text;
    b.addEventListener("click", action.run);
    div.append(b);
  }
  $("#messages").append(div);
  if (role === "assistant") voice.speak(text);
  div.scrollIntoView({ block: "nearest" });
  return div;
}
function showAuth() {
  state.user = null;
  $("#workspace").classList.add("hidden");
  $("#auth").classList.remove("hidden");
  $("#auth-title").textContent = state.setup
    ? "Your workspace starts here."
    : "Welcome back.";
  $("#auth-note").textContent = state.setup
    ? "Create the administrator account for this server. Use at least 12 characters for your password."
    : "Sign in to your annotation workspace.";
  $("#login-button").textContent = state.setup ? "Create workspace" : "Sign in";
}
async function start() {
  state.setup = (await api("/auth/status")).setup_required;
  if (state.setup) return showAuth();
  try {
    state.user = await api("/auth/me");
  } catch {
    return showAuth();
  }
  $("#auth").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#identity").textContent = state.user.username;
  $("#identity").title = state.user.username;
  $("#account-role").textContent = state.user.is_admin ? "Administrator" : "";
  $("#account-role").classList.toggle("hidden", !state.user.is_admin);
  const destination = new URLSearchParams(location.search);
  await loadProjects(
    destination.get("project") || sessionStorage.getItem("project"),
  );
  if (["datasets", "review"].includes(destination.get("page")))
    await returnToWorkList(destination.get("project"), destination.get("page"));
}
async function returnToWorkList(projectId, page) {
  if ($("#dialog").open)
    throw new Error("An unfinished form is open in this workspace.");
  if (!state.user || !["datasets", "review"].includes(page))
    throw new Error("Sign in to return to your project.");
  if (state.project?.id !== projectId) await loadProjects(projectId);
  if (state.project?.id !== projectId)
    throw new Error("This project is no longer available.");
  state.page = page;
  state.reviewFilter = "pending";
  state.datasetFilter = "all";
  state.searches[page] = "";
  state.pages[page] = 1;
  delete state.context.asset_id;
  await refresh();
  notice(
    page === "review"
      ? "Review saved. Choose the next pending case."
      : "Annotation submitted for review. Choose your next sample.",
  );
}
if (viewerReturns)
  viewerReturns.onmessage = async ({ data }) => {
    if (
      data?.type !== "return" ||
      typeof data.projectId !== "string" ||
      typeof data.requestId !== "string"
    )
      return;
    try {
      await returnToWorkList(data.projectId, data.page);
      window.focus();
      viewerReturns.postMessage({
        type: "returned",
        requestId: data.requestId,
      });
    } catch {
      viewerReturns.postMessage({
        type: "unavailable",
        requestId: data.requestId,
      });
    }
  };
async function loadProjects(id) {
  state.projects = await api("/projects");
  $("#project-select").innerHTML =
    '<option value="">Select a project</option>' +
    state.projects
      .map((p) => `<option value="${p.id}">${escapeHTML(p.name)}</option>`)
      .join("");
  await selectProject(
    state.projects.some((p) => p.id === id) ? id : state.projects[0]?.id,
  );
}
async function selectProject(id) {
  voice.cancel();
  state.project = state.projects.find((p) => p.id === id) ?? null;
  state.changeVersion = null;
  state.context = {};
  state.selectedFiles.clear();
  state.pages = { datasets: 1, review: 1, activity: 1 };
  state.searches = { datasets: "", review: "", activity: "" };
  state.evaluations = [];
  state.evaluationSets = [];
  state.modelSplits = [];
  state.evaluationVersions = [];
  state.selectedReviews.clear();
  state.datasetFilter = "all";
  state.members = [];
  state.assets = [];
  state.videos = [];
  state.dicomSeries = [];
  state.models = [];
  state.learners = [];
  state.recipes = [];
  state.jobs = [];
  state.decisions = [];
  state.credentials = [];
  state.roles = [];
  $("#project-select").value = state.project?.id ?? "";
  sessionStorage.setItem("project", state.project?.id ?? "");
  await refresh();
}
async function refresh({ automatic = false } = {}) {
  const coordinator = await api("/assistant/status");
  $(".chat-footnote").textContent = `Assistant ${coordinator.state}`;
  $(".chat-footnote").title = `${coordinator.model} · ${coordinator.message}`;
  if (state.project) {
    const prefix = `/projects/${state.project.id}`;
    // Read the counter first: a concurrent write will trigger another refresh.
    const change = await api(`${prefix}/changes`);
    const [
      project,
      assets,
      models,
      jobs,
      decisions,
      permissions,
      learners,
      recipes,
      dicomSeries,
      evaluations,
      evaluationSets,
      modelSplits,
      evaluationVersions,
      videos,
      videoCapabilities,
      reviewUnits,
    ] = await Promise.all([
      api(prefix),
      api(`${prefix}/assets`),
      api(`${prefix}/models`),
      api(`${prefix}/jobs`),
      api(`${prefix}/decisions`),
      api(`${prefix}/permissions`),
      api(`${prefix}/learners`),
      api(`${prefix}/recipes`),
      api(`${prefix}/dicom-series`),
      api(`${prefix}/evaluations`),
      api(`${prefix}/evaluation-sets`),
      api(`${prefix}/model-splits`),
      api(`${prefix}/evaluation-set-versions`),
      api(`${prefix}/videos`),
      api(`${prefix}/video-capabilities`),
      api(`${prefix}/review-units`),
    ]);
    const [credentials, members] = permissions.roles.includes("manager")
      ? await Promise.all([
          api(`${prefix}/credentials`),
          api(`${prefix}/members`),
        ])
      : [[], []];
    if (state.project?.id !== project.id) return;
    if (automatic && workspaceEditing()) return;
    if (state.changeVersion !== null && state.changeVersion > change.version)
      return;
    state.projects = state.projects.map((item) =>
      item.id === project.id ? project : item,
    );
    const option = Array.from($("#project-select").options).find(
      (item) => item.value === project.id,
    );
    if (option) option.textContent = project.name;
    Object.assign(state, {
      project,
      assets,
      models,
      jobs,
      decisions,
      credentials,
      members,
      roles: permissions.roles,
      changeVersion: change.version,
      evaluations,
      evaluationSets,
      modelSplits,
      evaluationVersions,
      learners,
      recipes,
      dicomSeries,
      videos,
      videoCapabilities,
      reviewUnits,
    });
  }
  render();
}
function render() {
  $("#project-select").title = state.project?.name || "Select a project";
  // Data refreshes keep the URL; section changes add one browser history entry.
  const destination = pageURL(state.page, location.href, state.project?.id);
  if (destination.href !== location.href)
    history[replaceNavigation ? "replaceState" : "pushState"](
      null,
      "",
      destination,
    );
  replaceNavigation = false;
  document.title = `${titles[state.page]} · MONAI Label`;
  const existing = new Set(samples(state).map((a) => a.id));
  state.selectedFiles = new Set(
    [...state.selectedFiles].filter((id) => existing.has(id)),
  );
  state.selectedReviews = new Set(
    [...state.selectedReviews].filter((id) =>
      reviewItems(state).some((a) => a.annotation_id === id),
    ),
  );
  for (const id of ["edit-project", "delete-project"])
    $(`#${id}`).classList.toggle("hidden", !state.project || !canManage());
  $("#page-title").textContent = titles[state.page];
  $("#breadcrumb").textContent = state.project?.name ?? "YOUR WORKSPACE";
  $("#chat-context").textContent = state.project
    ? `${state.project.name}${selectedAsset() ? ` / ${selectedAsset().name}` : ""}`
    : "No project selected";
  $("#asset-count").textContent =
    state.assets.length + state.videos.length || "";
  $("#review-count").textContent =
    [...state.assets, ...state.videos].filter((a) => statusOf(a) === "pending")
      .length || "";
  document
    .querySelectorAll("nav button")
    .forEach((b) =>
      b.classList.toggle("active", b.dataset.page === state.page),
    );
  const pages = {
    overview,
    datasets: () => datasets(state, canManage(), statusOf),
    models: () => modelLibrary(state, canManage, latestDecision),
    review: () => reviewQueue(state, statusOf, latestDecision),
    activity: () =>
      activity(state, canManage() || state.roles.includes("annotator")),
    team: () => team(state, canManage()),
  };
  $("#content").innerHTML =
    !state.project && state.page !== "overview"
      ? requireProject()
      : pages[state.page]();
  const selectReviews = $("#select-page-reviews");
  if (selectReviews) {
    const checks = [...document.querySelectorAll("[data-review-selection]")];
    const count = checks.filter((input) => input.checked).length;
    selectReviews.checked = count > 0 && count === checks.length;
    selectReviews.indeterminate = count > 0 && count < checks.length;
  }
  const selectAll = $("#select-all-files");
  if (selectAll) {
    const visible = pageSlice(
      visibleAssets(state, statusOf),
      state,
      "datasets",
    );
    const count = visible.filter((a) => state.selectedFiles.has(a.id)).length;
    selectAll.checked = count === visible.length && count > 0;
    selectAll.indeterminate = count > 0 && count < visible.length;
  }
}
function overview() {
  if (!state.project)
    return `<section class="hero"><h2>Start an annotation project</h2><p>Import images, choose a model and annotate in your preferred viewer.</p>${button("Create project", "project", "", "primary")}${button("Explore synthetic demo", "seed")}</section>`;
  const all = samples(state);
  const pending = all.filter((a) => statusOf(a) === "pending").length;
  const accepted = all.filter((a) => statusOf(a) === "accepted").length;
  const links = [
    ["datasets", "Samples", all.length],
    ["review", "Pending review", pending],
    ["review", "Accepted", accepted],
  ];
  return `<div class="stats">${links.map(([page, label, count]) => `<button class="stat" data-page="${page}" ${label === "Accepted" ? 'data-review-filter="accepted"' : label === "Pending review" ? 'data-review-filter="pending"' : ""}><small>${label}</small><strong>${count}</strong></button>`).join("")}</div><section class="card"><h2>Continue your work</h2><p class="muted">Open a sample to annotate, or review a submitted case.</p><div class="toolbar"><button class="primary" data-page="datasets">${actionLabel("Open datasets", "library")}</button><button data-page="review">${actionLabel("Review annotations", "check")}</button>${canManage() ? button("Import files", "dataset") : ""}</div></section>${state.project.instructions ? `<section class="card"><h3>Instructions for annotators (optional)</h3><p class="preserve-lines">${escapeHTML(state.project.instructions)}</p></section>` : ""}<details class="card" open><summary>Getting started</summary><ol class="getting-started"><li><strong>Import data.</strong> NIfTI, DICOM series, bounded pathology images or videos.</li><li><strong>Choose a model.</strong> Connect a hosted service or use a local segmentation model.</li><li><strong>Annotate and review.</strong> Prompt and edit in Slicer, QuPath, OHIF or CVAT; submit and review the annotated coverage.</li><li><strong>Train and compare.</strong> Train from accepted annotations; add independent evaluation data or a model-specific split when ready.</li></ol></details>${state.project.is_demo ? '<p class="muted">Synthetic demonstration data and CPU baseline models.</p>' : ""}`;
}
function requireProject() {
  return '<div class="empty"><h3>Select or create a project</h3><p>Your datasets, models, and team will appear here.</p></div>';
}
function field(label, name, type = "text", attrs = "", hint = "") {
  return `<label><span id="field-${name}-label">${label}</span><input name="${name}" type="${type}" aria-labelledby="field-${name}-label" ${hint ? `aria-describedby="field-${name}-hint"` : ""} ${attrs}>${hint ? `<small id="field-${name}-hint">${hint}</small>` : ""}</label>`;
}
function selectField(label, name, options) {
  return `<label><span id="field-${name}-label">${label}</span><select name="${name}" aria-labelledby="field-${name}-label">${options}</select></label>`;
}
function modal(title, html, submit, label = "Save") {
  $("#dialog-title").textContent = title;
  const form = $("#action-form");
  form.innerHTML =
    html +
    `<p class="form-error" role="alert"></p><button class="primary" type="submit">${submitLabel(label)}</button>`;
  form.onsubmit = async (event) => {
    event.preventDefault();
    const b = form.querySelector('button[type="submit"]');
    form.querySelector(".form-error").textContent = "";
    b.disabled = true;
    const controls = [...form.querySelectorAll('button[type="button"]')].map(
      (button) => [button, button.disabled],
    );
    controls.forEach(([button]) => (button.disabled = true));
    $("#close-dialog").disabled = true;
    $("#dialog").dataset.busy = "true";
    try {
      const close = await submit(new FormData(form));
      try {
        // Keep the old revision's actions inaccessible until the view is current.
        await refresh();
      } finally {
        if (close !== false) $("#dialog").close();
      }
    } catch (error) {
      form.querySelector(".form-error").textContent = error.message;
    } finally {
      b.disabled = false;
      controls.forEach(([button, disabled]) => (button.disabled = disabled));
      $("#close-dialog").disabled = false;
      delete $("#dialog").dataset.busy;
    }
  };
  if (!$("#dialog").open) $("#dialog").showModal();
}
async function openForm(kind, id) {
  if (kind !== "project" && !state.project)
    throw new Error("Select a project first.");
  const prefix = `/projects/${state.project?.id}`;
  if (kind === "video-import")
    return videoAction(kind, id, {
      state,
      api,
      modal,
      field,
      selectField,
      message,
      watch,
      safely,
    });
  if (kind === "project")
    return modal(
      "Create a project",
      field(
        "Project name",
        "name",
        "text",
        "required placeholder='Abdominal segmentation'",
      ) + field("Instructions for annotators (optional)", "instructions"),
      async (f) => {
        const p = await api("/projects", "POST", {
          name: f.get("name"),
          instructions: f.get("instructions"),
        });
        await loadProjects(p.id);
        message(
          `Created ${p.name}. Next, import files or choose an annotation model.`,
        );
      },
      "Create project",
    );
  if (kind === "dicom") {
    const projectId = state.project.id;
    return openDicomImport({
      api,
      projectId,
      onImport: async (job) => {
        state.page = "activity";
        await refresh();
        safely(() => watch(job.id, projectId));
      },
    });
  }
  if (kind === "dataset")
    return importFiles({ state, api, modal, field, selectField, message });
  if (kind === "split") {
    const asset = state.assets.find((a) => a.id === id);
    return modal(
      "Choose how to use these samples",
      `<p>${escapeHTML(asset.name)}</p><p class="muted">Applies to all samples from the same patient or slide: ${escapeHTML(asset.group_id)}. Evaluation samples stay separate from training.</p>` +
        selectField(
          "Use in this project",
          "split",
          '<option value="train">Training</option><option value="validation">Evaluation</option>',
        ),
      async (f) => {
        await api(`/assets/${id}/assign-${f.get("split")}`, "POST");
      },
      "Save selection",
    );
  }
  if (kind === "credential")
    return modal(
      "API keys",
      selectField(
        "API key to save",
        "id",
        '<option value="">Add a new key</option>' +
          state.credentials
            .map(
              (c) => `<option value="${c.id}">${escapeHTML(c.name)}</option>`,
            )
            .join(""),
      ) +
        field(
          "Key name",
          "name",
          "text",
          "required placeholder='e.g. Research account'",
        ) +
        field(
          "API key",
          "key",
          "password",
          "required autocomplete='off'",
          "Stored encrypted on this server. Existing keys are never sent back to your browser.",
        ),
      async (f) => {
        await api(`${prefix}/credentials`, "POST", {
          name: f.get("name"),
          api_key: f.get("key"),
          credential_id: f.get("id") || null,
        });
      },
    );
  if (kind === "model")
    return setupModel(
      { state, api, modal, field, selectField, escapeHTML, message },
      id,
    );
  if (kind === "model-provider")
    return importHostedModel(
      { state, api, modal, field, selectField, escapeHTML, message },
      "",
      state.models.find((m) => m.id === id),
    );
  if (kind === "derive-model") {
    const recipe = state.recipes.find((item) => item.id === "vista3d");
    let chosen = () => [];
    modal(
      "Create a VISTA3D project model",
      field(
        "Model name",
        "name",
        "text",
        'required maxlength="120" placeholder="e.g. Spleen specialist"',
      ) +
        `<p class="muted">Use this name in chat, for example “train Spleen specialist”.</p><p>Inherits all ${recipe?.supported_targets?.length || "supported"} base structures. Optionally choose initial training organs below; you can change them when starting fine-tuning.</p><div id="vista-structure-picker"></div>`,
      async (f) => {
        const name = f.get("name").trim();
        if (!name) throw new Error("Give this model a name to use in chat.");
        if (
          state.learners.some(
            (l) => l.name.trim().toLowerCase() === name.toLowerCase(),
          )
        )
          throw new Error(
            "A training setup with this name already exists. Choose a different name.",
          );
        const targets = chosen();
        if (!targets.length) {
          const learner = await api(`${prefix}/learners`, "POST", {
            name,
            recipe: "vista3d",
            initial_model_id: id,
            inherit_targets: true,
          });
          state.context.learner_id = learner.id;
          state.modelsTab = "training";
          message(
            `Created ${name} with inherited targets. Choose annotated organs when starting fine-tuning. You can refer to this model by name in chat.`,
          );
          return;
        }
        const reply = await api(`${prefix}/assistant`, "POST", {
          message:
            "Create a VISTA3D project model named exactly " +
            JSON.stringify(name) +
            " for fine-tuning from the selected base model, with these initial training structures: " +
            targets.join(", ") +
            ". Do not start training.",
          context: { model_id: id },
        });
        updateContext(reply.data);
        if (reply.data.project) state.project = reply.data.project;
        state.modelsTab = "training";
        message(reply.message);
      },
      "Create model",
    );
    chosen = structurePicker($("#vista-structure-picker"), recipe);
    return;
  }
  if (modelAction(kind, id, { state, modal, field, api, escapeHTML, message }))
    return;
  if (kind === "dataset-template")
    return importDatasetTemplate({
      state,
      api,
      modal,
      field,
      selectField,
      watch,
      safely,
    });
  if (kind === "start-training") {
    state.context.learner_id = id;
    const saved = state.learners.find((l) => l.id === id);
    const learner = {
      ...saved,
      config: {
        ...state.recipes.find((r) => r.id === saved.recipe)?.default_config,
        ...saved.config,
      },
    };
    const parents = state.models.filter(
      (m) =>
        m.provider === learner.recipe &&
        (((learner.inherit_targets ||
          m.label_ids.join() === learner.label_ids.join()) &&
          m.state_key) ||
          m.id === learner.initial_model_id),
    );
    modal(
      `Train ${learner.name}`,
      selectField(
        "How to start training",
        "mode",
        '<option value="scratch">Start a new model</option>' +
          (parents.length
            ? '<option value="fine_tune">Improve an existing model (fine-tune)</option><option value="continue">Resume a previous training run</option>'
            : ""),
      ) +
        '<div id="training-parent"></div>' +
        trainingSettings(learner) +
        '<div id="training-evaluation"></div>' +
        (learner.recipe === "vista3d"
          ? '<fieldset><legend>Structures to train</legend><div class="checkboxes">' +
            state.project.labels
              .filter(
                (label) =>
                  label.id &&
                  (
                    state.recipes.find((recipe) => recipe.id === learner.recipe)
                      ?.supported_targets || []
                  ).includes(label.name.toLowerCase()) &&
                  (learner.inherit_targets ||
                    learner.label_ids.includes(label.id)),
              )
              .map(
                (label) =>
                  `<label><input type="checkbox" name="targets" value="${label.id}" ${learner.label_ids.includes(label.id) ? "checked" : ""}>${escapeHTML(label.name)}</label>`,
              )
              .join("") +
            "</div></fieldset>"
          : "") +
        trainingSamples(),
      async (f) => {
        const targets =
          learner.recipe === "vista3d"
            ? [0, ...f.getAll("targets").map(Number)]
            : learner.label_ids;
        if (targets.length < 2)
          throw new Error("Choose at least one annotated organ to fine-tune.");
        const projectId = state.project.id;
        const job = await api(`${prefix}/learners/${id}/train`, "POST", {
          mode: f.get("mode"),
          parent_model_id: f.get("parent") || null,
          ...trainingEvaluationRequest(f),
          ...trainingSampleRequest(f),
          label_ids: targets,
          config: trainingOverrides(f, learner),
        });
        state.context.learner_id = id;
        safely(() => watch(job.id, projectId));
        state.page = "activity";
      },
      "Start training",
    );
    const form = $("#action-form");
    bindTrainingSettings(form);
    bindTrainingSamples(form, { state, learner, latestDecision });
    form.elements.mode.addEventListener("change", () => {
      $("#training-parent").innerHTML =
        form.elements.mode.value === "scratch"
          ? ""
          : selectField(
              "Model to start from",
              "parent",
              parents
                .map(
                  (m) =>
                    `<option value="${m.id}">${escapeHTML(m.name)} · ${new Date(m.created_at).toLocaleString()}</option>`,
                )
                .join(""),
            );
    });
    if (learner.initial_model_id) {
      form.elements.mode.value = "fine_tune";
      form.elements.mode.dispatchEvent(new Event("change"));
      form.elements.parent.value = learner.initial_model_id;
    }
    trainingEvaluation(form, { state, learner });
    return;
  }
  if (kind === "accept" || kind === "changes") {
    const a = state.assets.find((a) => a.id === id);
    return modal(
      kind === "accept" ? "Accept this revision" : "Request changes",
      `<p>${escapeHTML(a.name)} · revision ${a.revision}</p>` +
        field(
          kind === "changes"
            ? "What needs to change?"
            : "Review comment (optional)",
          "comment",
          "text",
          kind === "changes" ? "required" : "",
        ),
      async (f) => {
        await api(`/annotations/${a.annotation_id}/decision`, "POST", {
          verdict: kind === "accept" ? "accepted" : "changes_requested",
          comment: f.get("comment"),
        });
      },
      kind === "accept" ? "Accept revision" : "Request changes",
    );
  }
  if (kind === "team") {
    state.page = "team";
    return render();
  }
  if (kind === "user")
    return modal(
      "Create user",
      field("Username", "username", "text", "required") +
        field(
          "Password (at least 12 characters)",
          "password",
          "password",
          "required minlength='12' autocomplete='new-password'",
        ),
      async (f) => {
        const u = await api("/auth/users", "POST", {
          username: f.get("username"),
          password: f.get("password"),
        });
        message(
          `Created ${u.username}. Choose Assign roles to add them to this project.`,
        );
      },
    );
  if (kind === "member") {
    const users = state.user.is_admin ? await api("/auth/users") : [];
    const input = users.length
      ? selectField(
          "User",
          "user",
          users
            .map(
              (u) =>
                `<option value="${u.id}">${escapeHTML(u.username)}</option>`,
            )
            .join(""),
        )
      : id
        ? `<input type="hidden" name="user" value="${escapeHTML(id)}"><p>${escapeHTML(state.members.find((m) => m.user_id === id)?.username || "Project member")}</p>`
        : field(
            "Username",
            "username",
            "text",
            "required",
            "The account must already exist on this server.",
          );
    modal(
      id ? "Edit project roles" : "Assign project roles",
      input +
        '<fieldset><legend>Project access</legend><div class="checkboxes">' +
        ["manager", "annotator", "reviewer"]
          .map(
            (r) =>
              `<label><input type="checkbox" name="roles" value="${r}">${r[0].toUpperCase() + r.slice(1)}</label>`,
          )
          .join("") +
        "</div><small>Managers manage the project. Annotators create annotations. Reviewers check and accept them.</small></fieldset>",
      async (f) => {
        await api(`${prefix}/members`, "PUT", {
          user_id: f.get("user") || null,
          username: f.get("username") || null,
          roles: f.getAll("roles"),
        });
      },
    );
    if (id) {
      const form = $("#action-form");
      form.elements.user.value = id;
      const member = state.members.find((m) => m.user_id === id);
      form.querySelectorAll('[name="roles"]').forEach((input) => {
        input.checked = member.roles.includes(input.value);
      });
    }
    return;
  }
}
function updateContext(data) {
  for (const key of ["model_id", "learner_id", "snapshot_id", "evaluation_id"])
    if (data[key]) state.context[key] = data[key];
}
async function watch(jobId, originProject, viewerTab = null) {
  try {
    await watchJob(jobId, originProject, viewerTab);
  } catch (error) {
    closePendingViewer(viewerTab);
    throw error;
  }
}
async function watchJob(jobId, originProject, viewerTab) {
  const bubble = message("Job queued. You can continue using the workspace.");
  let job;
  do {
    await new Promise((r) => setTimeout(r, 700));
    job = await api(`/jobs/${jobId}`);
    bubble.querySelector("p").textContent =
      `${jobName(job.kind)} · ${job.status} · ${Math.round(job.progress * 100)}% ${job.progress_message || ""}`;
  } while (
    !["succeeded", "failed", "cancelled", "interrupted"].includes(job.status)
  );
  if (job.status !== "succeeded") {
    if (job.kind === "batch_annotate") {
      message(
        `${job.result?.annotation_ids?.length || 0} annotations submitted before the batch stopped.`,
      );
      for (const failure of job.result?.failed || [])
        message(`${failure.name}: ${failure.error}`);
    }
    throw new Error(job.error || `Job ${job.status}`);
  }
  if (state.project?.id === originProject) {
    updateContext(job.result);
    await refresh();
  }
  if (job.result.url) {
    const viewerName =
      job.kind === "video_editor"
        ? "CVAT"
        : { slicer: "3D Slicer", qupath: "QuPath" }[job.result.viewer] ||
          "OHIF";
    const viewerUrl = new URL(job.result.url, location.origin);
    if (viewerName === "OHIF")
      viewerUrl.searchParams.set("workspace", workspaceId);
    if (openPreparedViewer(viewerUrl.href, viewerTab, viewerName !== "OHIF")) {
      message(`${viewerName} opened${viewerTab ? " in a new tab" : ""}.`);
      return;
    }
    const bubble = message(`${viewerName} is ready. Open the selected sample:`);
    const link = document.createElement("a");
    link.href = viewerUrl.href;
    link.textContent = `Open ${viewerName} annotation viewer ↗`;
    link.target = "_blank";
    link.rel = "noopener";
    bubble.append(link);
    return;
  }
  if (job.kind === "batch_annotate") {
    message(
      `Segmentation finished: ${job.result.asset_ids.length} of ${job.result.selected_count} images segmented, ${job.result.annotation_ids.length} submitted to Reviews as pending, ${job.result.failed.length} failed.`,
    );
    for (const failure of job.result.failed)
      message(`${failure.name}: ${failure.error}`);
    return;
  }
  if (job.kind === "select") {
    message(
      `Selected ${job.result.items.length} cases. Selection alone does not run segmentation or submit reviews.`,
    );
    return;
  }
  if (job.kind === "dataset_import") {
    if (job.result.video_ids?.length) {
      message(
        `Video import finished: ${job.result.video_ids.length} clip available in Datasets. Open CVAT to annotate, then submit the saved tracks for review.`,
      );
      return;
    }
    message(
      `Dataset import finished: ${job.result.asset_ids.length} cases available, ${job.result.annotation_ids.length} reference masks pending review, ${job.result.failed.length} failed.`,
    );
    for (const failure of job.result.failed)
      message(`${failure.case}: ${failure.error}`);
    return;
  }
  if (job.kind === "dicom_import" && job.result.asset_ids) {
    message(
      `DICOM import finished: ${job.result.asset_ids.length} imported, ${job.result.skipped_asset_ids.length} already present, ${job.result.failed.length} failed.`,
    );
    for (const failure of job.result.failed)
      message(`${failure.series_uid}: ${failure.error}`);
    return;
  }
  message(
    job.kind === "viewer"
      ? "Viewer launch requested. It will open the selected sample with the annotation assistant. Check Activity for its local launch log."
      : `${jobName(job.kind)} completed. Results are available in Models and Activity.`,
  );
}
async function launchViewer(
  id,
  name,
  mode = state.page === "review" ? "review" : "annotation",
) {
  if (!id) throw new Error("Select a sample first.");
  const target = desktopLaunchTarget;
  const viewerTab =
    name === "ohif" || target === "browser" ? reserveViewerTab() : null;
  state.context.asset_id = id;
  render();
  const project = state.project.id;
  try {
    const job = await api(
      `/assets/${id}/viewer?mode=${mode}&target=${target}${name ? "&name=" + name : ""}`,
      "POST",
    );
    message(
      `Preparing the viewer ${target === "browser" || name === "ohif" ? "in your browser" : "on this computer"}. The first launch may take several minutes; later launches reuse it.`,
    );
    await watch(job.id, project, viewerTab);
  } catch (error) {
    closePendingViewer(viewerTab);
    throw error;
  }
}
async function send(text) {
  if (state.chatBusy) return;
  state.chatBusy = true;
  $('#chat-form [type="submit"]').disabled = true;
  $("#new-conversation").disabled = true;
  try {
    await sendPrompt(text);
  } finally {
    state.chatBusy = false;
    $('#chat-form [type="submit"]').disabled = false;
    $("#new-conversation").disabled = false;
  }
}
async function sendPrompt(text) {
  message(text, "user");
  const projectId = state.project?.id ?? null;
  const conversationKey = `${projectId || "workspace"}:${state.context.asset_id || ""}`;
  const reply = await api("/assistant", "POST", {
    message: text,
    project_id: projectId,
    context: state.context,
    conversation_id: state.conversations.get(conversationKey) || null,
    request_id: randomId(),
  });
  state.conversations.set(conversationKey, reply.conversation_id);
  message(reply.message);
  if (reply.data.select_project) await loadProjects(reply.data.select_project);
  if (state.project?.id !== projectId && !reply.data.select_project) return;
  updateContext(reply.data);
  if (reply.data.project && reply.data.project.id === state.project?.id) {
    state.project = reply.data.project;
  }
  if (["models", "datasets", "review"].includes(reply.data.page)) {
    state.page = reply.data.page;
    await refresh();
  }
  if (reply.data.form) await openForm(reply.data.form);
  if (reply.data.client_action === "use_viewer")
    await launchViewer(
      reply.data.asset_id || state.context.asset_id,
      reply.data.viewer,
    );
  if (reply.job_id) {
    safely(() => watch(reply.job_id, projectId));
    await refresh();
  } else render();
}
async function action(name, id) {
  if (name.startsWith("video-"))
    return videoAction(name, id, {
      state,
      api,
      modal,
      field,
      selectField,
      message,
      watch,
      safely,
    });
  if (name === "select-filtered-reviews") {
    visibleReviews(state, statusOf).forEach((a) =>
      state.selectedReviews.add(a.annotation_id),
    );
    return render();
  }
  if (name === "evaluation-sets" || name.startsWith("evaluation-set-"))
    return evaluationSetAction(name, id, {
      state,
      api,
      modal,
      field,
      refresh,
      render,
      notice,
      latestDecision,
    });
  if (name === "clear-review-selection") {
    state.selectedReviews.clear();
    return render();
  }
  notice("");
  if (name === "model-targets") return showModelTargets(state, id, modal);
  if (name === "default-model") {
    await api(`/projects/${state.project.id}/annotation-model`, "PUT", {
      model_id: id,
      base_version: state.project.version,
    });
    await refresh();
    notice(
      `${state.models.find((model) => model.id === id).name} is now the project default.`,
    );
    return;
  }
  if (["snapshot", "evaluate", "promote"].includes(name)) {
    return learningAction(name, id, {
      state,
      api,
      modal,
      escapeHTML,
      message,
      render,
      watch,
      safely,
      refresh,
    });
  }
  if (name === "edit-project") {
    const project = state.project;
    return modal(
      "Edit project",
      field(
        "Project name",
        "name",
        "text",
        `required maxlength="120" value="${escapeHTML(project.name)}"`,
      ) +
        `<label>Instructions for annotators (optional)<textarea name="instructions" rows="4" maxlength="8000">${escapeHTML(project.instructions)}</textarea></label>`,
      async (form) => {
        await api(`/projects/${project.id}`, "PATCH", {
          name: form.get("name").trim(),
          instructions: form.get("instructions"),
          base_version: project.version,
        });
        notice("Project settings saved.");
      },
      "Save changes",
    );
  }
  if (name === "delete-project" || name === "delete-files") {
    if (!canManage()) throw new Error("Project manager access is required.");
    if (state.chatBusy)
      throw new Error(
        "Wait for the current prompt to finish before deleting data.",
      );
    const project = state.project;
    const assets = samples(state).filter((a) =>
      id ? a.id === id : state.selectedFiles.has(a.id),
    );
    const finished = async () => {
      for (const key of state.conversations.keys()) {
        if (
          name === "delete-project"
            ? key.startsWith(`${project.id}:`)
            : assets.some((a) => key === `${project.id}:${a.id}`)
        )
          state.conversations.delete(key);
      }
      if (name === "delete-project") {
        state.page = "overview";
        await loadProjects();
        notice(`Deleted project ${project.name}.`);
      } else {
        state.selectedFiles.clear();
        if (assets.some((a) => a.id === state.context.asset_id))
          state.context = {};
        await refresh();
        notice(
          `Deleted ${assets.length} file${assets.length === 1 ? "" : "s"}.`,
        );
      }
    };
    return (name === "delete-project" ? deleteProject : deleteFiles)({
      project,
      assets,
      api,
      modal,
      field,
      escapeHTML,
      finished,
    });
  }
  if (
    [
      "project",
      "dataset",
      "dicom",
      "split",
      "model",
      "credential",
      "team",
      "user",
      "member",
      "start-training",
      "rename-learner",
      "delete-learner",
      "rename-model",
      "model-provider",
      "delete-model",
      "dataset-template",
      "derive-model",
      "accept",
      "changes",
    ].includes(name)
  )
    return openForm(name, id);
  if (name === "select") {
    state.context.asset_id = id;
    render();
    return message(
      `Selected ${selectedAsset().name}. Ask “view in ${selectedAsset().kind === "image2d" ? "QuPath" : "3D Slicer"}” to open it${selectedAsset().kind === "volume3d" ? ", or choose OHIF" : ""}.`,
    );
  }
  if (name === "review-unit")
    return inspectReview(id, {
      state,
      modal,
      api,
      refresh,
      latestDecision,
      openViewer: (item) =>
        safely(() =>
          item.kind === "video"
            ? action("video-inspect", item.source_id)
            : launchViewer(item.source_id, "qupath", "annotation"),
        ),
    });
  if (name === "viewer") return launchViewer(id);
  if (name === "ohif") return launchViewer(id, "ohif");
  if (name === "seed") {
    const p = await api("/demo", "POST");
    await loadProjects(p.project_id);
    message(
      "Created 12 synthetic NIfTI samples and a fixed demo baseline. Open a sample in Slicer to annotate it, or run the CLI demo for an automated learning cycle.",
    );
    return;
  }

  if (name === "baseline") {
    state.context.baseline_id = id;
    render();
    message(
      `Evaluation baseline: ${state.models.find((m) => m.id === id).name}.`,
    );
    return;
  }
  if (name === "manage-training-model") {
    state.modelsTab = "training";
    render();
    document
      .querySelector(`[data-learner-id="${id}"]`)
      ?.scrollIntoView({ block: "center" });
    return;
  }
  if (name === "model-tab") {
    state.modelsTab = id;
    return render();
  }
  if (name === "sample-details") {
    const sample = samples(state).find((a) => a.id === id);
    if (!sample) throw new Error("This sample is no longer available.");
    modal(
      sample.name,
      `<dl class="details-list"><dt>Type</dt><dd>${sampleType(sample)}</dd><dt>Patient, slide or procedure group</dt><dd>${escapeHTML(sample.group_id)}</dd><dt>Dimensions</dt><dd>${escapeHTML(sampleDimensions(sample))}</dd><dt>Revision</dt><dd>${sample.revision}</dd><dt>Status</dt><dd>${badge(sampleUse(sample))} ${badge(statusOf(sample))}</dd>${sample.kind === "video" ? "<dt>Training and evaluation</dt><dd>Accepted polygon ranges can train a 2D segmentation model. Tracking metrics are not available.</dd>" : ""}</dl><div class="toolbar">${sample.kind === "video" ? button("Track data", "video-tracks", id) : ""}${sample.annotation_id && sample.kind === "volume3d" ? `<a class="download-link" href="/api/assets/${id}/segmentation.nii" download>${actionLabel("Download mask", "external")}</a>` : ""}${canManage() ? button("Delete file", "delete-files", id, "danger") : ""}</div>`,
      async () => true,
      "Close",
    );
    return;
  }
  if (name === "model-split") return showModelSplit({ state, modal }, id);
  if (name === "evaluation-import")
    return importFiles(
      { state, api, modal, field, selectField, message },
      "validation",
    );
  if (name === "job-details") {
    const job = state.jobs.find((j) => j.id === id);
    if (
      ["train", "evaluate", "training_report", "batch_annotate"].includes(
        job.kind,
      )
    )
      return trainingResults(job, { state, api, modal, canManage });
    modal(
      jobName(job.kind),
      `<p>${badge(job.status)} · ${escapeHTML(new Date(job.created_at).toLocaleString())}</p><p>${escapeHTML(job.error || job.progress_message || "")}</p><details><summary>Technical details</summary><pre class="code">${escapeHTML(JSON.stringify({ request: job.request, result: job.result }, null, 2))}</pre></details>`,
      async () => true,
      "Close",
    );
    return;
  }
  if (name === "page") {
    state.pages[state.page] = Number(id);
    return render();
  }
  if (name === "select-filtered") {
    visibleAssets(state, statusOf).forEach((a) =>
      state.selectedFiles.add(a.id),
    );
    return render();
  }
  if (name === "clear-selection") {
    state.selectedFiles.clear();
    return render();
  }
  if (name === "reset-filters") {
    state.searches[state.page] = "";
    if (state.page === "datasets") state.datasetFilter = "all";
    if (state.page === "activity") state.activityFilter = "all";
    if (state.page === "review") state.reviewFilter = "pending";
    state.pages[state.page] = 1;
    return render();
  }
  if (name === "cancel") {
    await api(`/jobs/${id}/cancel`, "POST");
    return refresh();
  }
}
function safely(work) {
  Promise.resolve()
    .then(work)
    .catch((error) => {
      notice(error.message);
      message(error.message);
    });
}
document.addEventListener("click", (event) => {
  const b = event.target.closest("button");
  if (!b) return;
  if (b.dataset.page) {
    if (b.dataset.reviewFilter) state.reviewFilter = b.dataset.reviewFilter;
    state.page = b.dataset.page;
    $("#sidebar-content").classList.remove("mobile-open");
    $("#mobile-nav").setAttribute("aria-expanded", "false");
    render();
  }
  if (b.dataset.action) safely(() => action(b.dataset.action, b.dataset.id));
  if (b.dataset.prompt) safely(() => send(b.dataset.prompt));
});
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const f = new FormData(event.target);
  $("#login-button").disabled = true;
  try {
    await api(state.setup ? "/auth/setup" : "/auth/login", "POST", {
      username: f.get("username"),
      password: f.get("password"),
    });
    event.target.reset();
    $("#auth-error").textContent = "";
    await start();
  } catch (error) {
    $("#auth-error").textContent = error.message;
  } finally {
    $("#login-button").disabled = false;
  }
});
$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.chatBusy) return;
  voice.cancel();
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  safely(() => send(text));
});
$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#project-select").addEventListener("change", (event) =>
  safely(() => selectProject(event.target.value)),
);
$("#refresh").addEventListener("click", () => {
  notice("");
  safely(refresh);
});
$("#new-project").addEventListener("click", () =>
  safely(() => openForm("project")),
);
$("#close-dialog").addEventListener("click", () => $("#dialog").close());
$("#dialog").addEventListener("cancel", (event) => {
  if ($("#dialog").dataset.busy) event.preventDefault();
});
$("#new-conversation").addEventListener("click", () => {
  voice.cancel();
  const key = `${state.project?.id || "workspace"}:${state.context.asset_id || ""}`;
  state.conversations.delete(key);
  $("#messages").replaceChildren();
  message("New conversation.");
});
$("#logout").addEventListener("click", () =>
  safely(async () => {
    await api("/auth/logout", "POST");
    location.reload();
  }),
);
safely(start);

document.addEventListener("change", (event) => {
  if (event.target.matches("[data-review-decision]")) {
    const input = event.target;
    const verdict = input.value;
    input.value = "";
    input.disabled = true;
    safely(async () => {
      try {
        await reviewAction(verdict, {
          state,
          api,
          refresh,
          notice,
          modal,
          latestDecision,
        });
      } finally {
        input.disabled = false;
      }
    });
  } else if (event.target.id === "select-page-reviews") {
    const checked = event.target.checked;
    document.querySelectorAll("[data-review-selection]").forEach((input) => {
      if (checked) state.selectedReviews.add(input.dataset.reviewSelection);
      else state.selectedReviews.delete(input.dataset.reviewSelection);
    });
    render();
  } else if (event.target.dataset.reviewSelection) {
    if (event.target.checked)
      state.selectedReviews.add(event.target.dataset.reviewSelection);
    else state.selectedReviews.delete(event.target.dataset.reviewSelection);
    render();
  } else if (event.target.id === "select-all-files") {
    pageSlice(visibleAssets(state, statusOf), state, "datasets").forEach(
      (a) => {
        if (event.target.checked) state.selectedFiles.add(a.id);
        else state.selectedFiles.delete(a.id);
      },
    );
    render();
  } else if (event.target.dataset.fileSelection) {
    if (event.target.checked)
      state.selectedFiles.add(event.target.dataset.fileSelection);
    else state.selectedFiles.delete(event.target.dataset.fileSelection);
    render();
  }
  if (event.target.dataset.context) {
    if (event.target.dataset.context === "evaluation_set_id") {
      delete state.context.evaluation_version_id;
      delete state.context.snapshot_id;
    }
    if (event.target.value)
      state.context[event.target.dataset.context] = event.target.value;
    else delete state.context[event.target.dataset.context];
    render();
  }
  const filters = {
    "review-filter": "reviewFilter",
    "dataset-filter": "datasetFilter",
    "activity-filter": "activityFilter",
  };
  if (filters[event.target.id]) {
    state[filters[event.target.id]] = event.target.value;
    state.pages[state.page] = 1;
    render();
  }
});
window.addEventListener("focus", () => updateIfChanged());
window.addEventListener("popstate", () =>
  safely(async () => {
    const destination = new URL(location.href);
    state.page = pageFromURL(destination);
    if (!state.user) return;
    const project = destination.searchParams.get("project");
    if (project && project !== state.project?.id) {
      replaceNavigation = true;
      await loadProjects(project);
    } else render();
  }),
);
document.addEventListener("visibilitychange", () => updateIfChanged());

document.addEventListener("input", (event) => {
  if (!event.target.dataset.search) return;
  const input = event.target;
  const position = input.selectionStart;
  state.searches[input.dataset.search] = input.value;
  state.pages[input.dataset.search] = 1;
  render();
  const replacement = document.querySelector(
    `[data-search="${input.dataset.search}"]`,
  );
  replacement.focus();
  replacement.setSelectionRange(position, position);
});
// One small server counter covers every viewer and other users' committed changes.
// Defer while editing a form/search; the unchanged local counter retries afterwards.
let polling = false;
function workspaceEditing() {
  return (
    $("#dialog").open ||
    document.activeElement?.closest(
      "#content input, #content select, #content textarea",
    )
  );
}
async function updateIfChanged() {
  if (
    polling ||
    document.hidden ||
    !state.user ||
    !state.project ||
    workspaceEditing()
  )
    return;
  polling = true;
  const projectId = state.project.id;
  try {
    const change = await api(`/projects/${projectId}/changes`);
    if (
      state.project?.id === projectId &&
      state.changeVersion !== change.version &&
      !workspaceEditing()
    ) {
      await refresh({ automatic: true });
    }
  } catch (error) {
    notice(error.message);
  } finally {
    polling = false;
  }
}
setInterval(updateIfChanged, 3000);

function showAssistant(open) {
  $("#workspace").classList.toggle("assistant-hidden", !open);
  $("#toggle-assistant").setAttribute("aria-expanded", String(open));
  $("#toggle-assistant").classList.toggle("active", open);
  if (open) $("#chat-input").focus();
}
$("#toggle-assistant").addEventListener("click", () =>
  showAssistant($("#workspace").classList.contains("assistant-hidden")),
);
$("#close-assistant").addEventListener("click", () => {
  showAssistant(false);
  $("#toggle-assistant").focus();
});
if (window.innerWidth < 1200) showAssistant(false);

$("#mobile-nav").addEventListener("click", () => {
  const open = $("#sidebar-content").classList.toggle("mobile-open");
  $("#mobile-nav").setAttribute("aria-expanded", String(open));
});

staticIcons();
