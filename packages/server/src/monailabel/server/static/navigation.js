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
