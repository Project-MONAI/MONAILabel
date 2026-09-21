import { watchDesktop } from "./desktop-lifecycle.js";

const identifier = location.pathname.split("/")[2];
const status = document.querySelector("#status");
const names = { slicer: "3D Slicer", qupath: "QuPath" };

if (identifier) {
  window.addEventListener("pagehide", () => {
    navigator.sendBeacon(`/api/desktops/${identifier}/close-tab`);
  });
}

async function request(path, method = "GET") {
  const response = await fetch(`/api/desktops${path}`, { method });
  if (response.status === 401)
    throw new Error("Sign in to MONAI Label, then reopen this session.");
  if (!response.ok)
    throw new Error(
      (await response.json()).detail || "The desktop is unavailable.",
    );
  return response.status === 204 ? null : response.json();
}

async function end(id) {
  if (
    !confirm(
      "End this desktop session? Submit your work first. Unsaved changes in the viewer will be lost.",
    )
  )
    return;
  await request(`/${id}`, "DELETE");
  location.assign("/desktop");
}

function report(error) {
  const message = document.querySelector("#error");
  message.textContent = error.message;
  message.hidden = false;
}

function fitDesktop(frame, viewer) {
  const viewport = document.querySelector("#viewport");
  // Native applications have minimum window sizes. On smaller screens, keep
  // their controls reachable while matching the browser's aspect ratio.
  const minimum = viewer === "slicer" ? [1440, 900] : [960, 800];
  const observer = new ResizeObserver(() => {
    const { width, height } = viewport.getBoundingClientRect();
    if (!width || !height) return;
    const scale = Math.min(1, width / minimum[0], height / minimum[1]);
    frame.style.width = `${Math.ceil(width / scale)}px`;
    frame.style.height = `${Math.ceil(height / scale)}px`;
    frame.style.transform = `scale(${scale})`;
  });
  observer.observe(viewport);
}

async function start() {
  if (identifier) {
    const info = await request(`/${identifier}`);
    document.title = `${names[info.viewer]} — MONAI Label`;
    const frame = document.querySelector("#display");
    document.querySelector("#viewport").hidden = false;
    fitDesktop(frame, info.viewer);
    const destination = info.mode === "review" ? "/reviews" : "/datasets";
    watchDesktop(
      frame,
      identifier,
      `${destination}?project=${info.project_id}`,
    );
    frame.src = `/desktop/${identifier}/client/vnc.html`;
  } else {
    document.querySelector("header").hidden = false;
    document.querySelector("#sessions").hidden = false;
    const sessions = await request("");
    if (!sessions.length)
      status.textContent =
        "No running sessions. Open Slicer or QuPath from Datasets.";
    for (const session of sessions) {
      const row = document.createElement("div");
      row.className = "session";
      const link = document.createElement(session.url ? "a" : "span");
      if (session.url) link.href = session.url;
      link.textContent = `${names[session.viewer]} · ${session.name} · ${session.mode}`;
      const button = document.createElement("button");
      button.textContent = "End session";
      button.onclick = () => end(session.id).catch(report);
      row.append(link, button);
      document.querySelector("#list").append(row);
    }
  }
}
start().catch(report);
