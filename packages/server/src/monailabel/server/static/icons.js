// Small shared action icons. Text remains the primary label for consequential actions.
const paths = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  edit: '<path d="m16 3 5 5-12 12-6 1 1-6Z M14 5l5 5"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
  server:
    '<rect x="3" y="3" width="18" height="7" rx="2"/><rect x="3" y="14" width="18" height="7" rx="2"/><path d="M7 6.5h.01M7 17.5h.01M15 6.5h3M15 17.5h3"/>',
  library: '<path d="M3 3v18M8 3v18M13 3v18m4-18 4 17"/>',
  external: '<path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7"/>',
  play: '<path d="m7 3 14 9-14 9Z"/>',
  key: '<circle cx="8" cy="8" r="5"/><path d="m12 12 9 9m-3-3 3-3m-6 0 3-3"/>',
  users:
    '<circle cx="9" cy="7" r="4"/><path d="M2 21v-3a7 7 0 0 1 14 0v3M17 3a4 4 0 0 1 0 8m2 4a5 5 0 0 1 3 4v2"/>',
  snapshot:
    '<rect x="3" y="3" width="14" height="14" rx="2"/><path d="M7 21h14V7M7 10h6M10 7v6"/>',
  compare:
    '<path d="M12 3v18M3 8h6M15 8h6M3 16h6M15 16h6m-9-3 3 3-3 3m12-14-3 3 3 3"/>',
  search: '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
  check: '<path d="m4 12 5 5L20 6"/>',
  left: '<path d="m15 5-7 7 7 7"/>',
  right: '<path d="m9 5 7 7-7 7"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  logout: '<path d="M9 3H3v18h6m6-14 5 5-5 5M8 12h12"/>',
};
export const icon = (name) =>
  `<svg class="action-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${paths[name] || ""}</svg>`;
export const actionIcons = {
  project: "plus",
  model: "plus",
  user: "plus",
  member: "users",
  "rename-model": "edit",
  "rename-learner": "edit",
  "edit-project": "edit",
  "delete-model": "trash",
  "delete-learner": "trash",
  "delete-files": "trash",
  "delete-project": "trash",
  dataset: "upload",
  "evaluation-import": "upload",
  dicom: "server",
  "dataset-template": "library",
  viewer: "external",
  ohif: "external",
  credential: "key",
  "start-training": "play",
  snapshot: "snapshot",
  "evaluation-sets": "library",
  "evaluation-set-create": "plus",
};
export function actionLabel(text, name) {
  return name ? `${icon(name)}<span>${text}</span>` : text;
}
// Dialog confirmations keep their full action text; Cancel, Close and choices stay plain.
export function submitLabel(text) {
  const name = {
    "Import files": "upload",
    "Import evaluation dataset": "upload",
    "Import dataset": "upload",
    "Import samples": "upload",
    "Import selected": "upload",
    "Import all matches": "upload",
    Connect: "server",
    "Connect model": "plus",
    "Create project": "plus",
    "Create model": "plus",
    "Create user": "plus",
    "Rename model": "edit",
    "Delete model": "trash",
    "Delete project": "trash",
    "Delete files": "trash",
    "Start training": "play",
    "Compare models": "compare",
    "Create snapshot": "snapshot",
    "Accept revision": "check",
  }[text];
  return actionLabel(text, name);
}
export function paginationButton(direction, attributes) {
  const label = direction === "left" ? "Previous page" : "Next page";
  return `<button type="button" class="icon-control" aria-label="${label}" title="${label}" ${attributes}>${icon(direction)}</button>`;
}
export function staticIcons() {
  for (const [id, name] of [
    ["new-project", "plus"],
    ["edit-project", "edit"],
    ["delete-project", "trash"],
    ["logout", "logout"],
    ["new-conversation", "plus"],
  ]) {
    const node = document.getElementById(id);
    if (node) node.innerHTML = actionLabel(node.textContent.trim(), name);
  }
  const close = document.getElementById("close-dialog");
  if (close) {
    close.innerHTML = icon("close");
    close.title = "Close dialog";
  }
}
