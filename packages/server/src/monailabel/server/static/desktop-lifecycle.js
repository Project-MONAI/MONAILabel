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

// A lost display connection does not mean that the native viewer has exited.
export function watchDesktop(frame, identifier, returnUrl) {
  let timer;
  let monitoring = false;
  let generation = 0;

  function stop() {
    monitoring = false;
    generation += 1;
    clearTimeout(timer);
  }

  async function check(current) {
    try {
      const response = await fetch(`/api/desktops/${identifier}`, {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      });
      if (current !== generation) return;
      if (response.status === 410) {
        stop();
        window.close();
        // Manually opened tabs may refuse close(); return to the workspace instead.
        setTimeout(() => location.replace(returnUrl), 100);
        return;
      }
      if (response.status === 401 || response.status === 403) {
        stop();
        return;
      }
    } catch {
      // A network interruption or backend restart must preserve the viewer tab.
    }
    if (monitoring && current === generation) {
      timer = setTimeout(() => check(current), 2000);
    }
  }

  window.addEventListener("message", (event) => {
    if (
      event.origin !== location.origin ||
      event.source !== frame.contentWindow
    )
      return;
    if (event.data?.type === "monailabel-desktop-connected") stop();
    if (event.data?.type === "monailabel-desktop-disconnected" && !monitoring) {
      monitoring = true;
      check(generation);
    }
  });
  window.addEventListener("pagehide", stop);
}
