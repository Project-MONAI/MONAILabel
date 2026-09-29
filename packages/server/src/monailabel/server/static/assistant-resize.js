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

const storageKey = "monailabel-assistant-width";

export function bindAssistantResize(workspace, panel, handle) {
  let preferred = null;
  try {
    const saved = Number(localStorage.getItem(storageKey));
    if (Number.isFinite(saved) && saved >= 300) preferred = saved;
  } catch {
    // Resizing also works when browser storage is unavailable.
  }
  const limits = () => {
    const width = window.innerWidth;
    const maximum =
      width >= 1200
        ? Math.min(960, width - (width >= 1550 ? 232 : 224) - 360)
        : width;
    return [Math.min(320, maximum), maximum];
  };
  const apply = () => {
    const [minimum, maximum] = limits();
    const fallback = window.innerWidth >= 1550 ? 380 : 354;
    const width = Math.round(
      Math.max(minimum, Math.min(maximum, preferred ?? fallback)),
    );
    workspace.style.setProperty("--assistant-width", `${width}px`);
    handle.setAttribute("aria-valuemin", String(minimum));
    handle.setAttribute("aria-valuemax", String(maximum));
    handle.setAttribute("aria-valuenow", String(width));
    handle.setAttribute("aria-valuetext", `${width} pixels`);
  };
  const save = () => {
    try {
      localStorage.setItem(storageKey, String(preferred));
    } catch {
      // Keep the current width for this tab.
    }
  };
  let drag = null;
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    handle.focus();
    drag = {
      id: event.pointerId,
      x: event.clientX,
      width: panel.getBoundingClientRect().width,
    };
    handle.setPointerCapture(event.pointerId);
    workspace.classList.add("resizing-assistant");
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const [minimum, maximum] = limits();
    preferred = Math.max(
      minimum,
      Math.min(maximum, drag.width + drag.x - event.clientX),
    );
    apply();
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    workspace.classList.remove("resizing-assistant");
    if (preferred !== null) save();
  };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  handle.addEventListener("lostpointercapture", end);
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const [minimum, maximum] = limits();
    const current = panel.getBoundingClientRect().width;
    preferred =
      event.key === "Home"
        ? minimum
        : event.key === "End"
          ? maximum
          : Math.max(
              minimum,
              Math.min(
                maximum,
                current + (event.key === "ArrowLeft" ? 32 : -32),
              ),
            );
    apply();
    save();
  });
  window.addEventListener("resize", apply);
  apply();
}
