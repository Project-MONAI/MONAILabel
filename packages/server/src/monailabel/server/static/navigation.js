// Workspace sections have real URLs; API and viewer routes remain separate.
const paths = {
  overview: "/",
  datasets: "/datasets",
  models: "/models",
  review: "/reviews",
  activity: "/activity",
  team: "/team",
};

export function pageFromURL(url) {
  const path = url.pathname.replace(/\/$/, "") || "/";
  // Older OHIF sessions return to /?project=…&page=datasets (or review).
  const legacy = url.searchParams.get("page");
  if (path === "/" && Object.hasOwn(paths, legacy)) return legacy;
  return Object.keys(paths).find((page) => paths[page] === path) || "overview";
}

export function pageURL(page, current, projectId) {
  const url = new URL(current);
  url.pathname = paths[page];
  url.searchParams.delete("page");
  // Keep explicit project links correct when the project changes. Ordinary
  // section URLs stay short; their project selection lives in session storage.
  if (url.searchParams.has("project")) {
    if (projectId) url.searchParams.set("project", projectId);
    else url.searchParams.delete("project");
  }
  return url;
}
