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

export function remainingLabel(seconds) {
  const value = Math.max(0, Math.ceil(seconds));
  return `${Math.floor(value / 3600).toString().padStart(2, "0")}:${Math.floor((value % 3600) / 60).toString().padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`;
}

export function demoSession({ banner, onExpired, surface = "workspace", trackActivity = true }) {
  let active = false;
  let recording = false;
  let timer = null;
  let deadline = null;
  let nextSync = 0;
  let syncing = false;
  let inFlight = 0;
  let lastEvent = -Infinity;
  let generation = 0;

  function stop() {
    generation++;
    active = recording = false;
    syncing = false;
    deadline = null;
    clearInterval(timer);
    timer = null;
    banner.hidden = true;
  }

  function expired() {
    stop();
    banner.hidden = false;
    banner.textContent = "Demo access has ended.";
    onExpired();
  }

  function apply(info) {
    recording = Boolean(info.activity_recording);
    // Base the display on server time and a monotonic clock, not laptop clock settings.
    deadline = info.expires_at
      ? performance.now() + Math.max(0, Date.parse(info.expires_at) - Date.parse(info.server_now))
      : null;
  }

  async function sync() {
    if (syncing || !active) return;
    syncing = true;
    const started = generation;
    try {
      const response = await fetch("/api/auth/session", { cache: "no-store" });
      if (!active || started !== generation) return;
      if (response.status === 401) return expired();
      if (response.ok) {
        const info = await response.json();
        if (active && started === generation) apply(info);
      }
    } catch {
      // No extension while offline: retain the previous server deadline.
    } finally {
      if (started === generation) {
        syncing = false;
        nextSync = performance.now() + 30000;
      }
    }
  }

  function tick() {
    if (!active) return;
    const left = deadline === null ? null : (deadline - performance.now()) / 1000;
    if (left !== null && left <= 0) return expired();
    banner.hidden = left === null && !recording;
    banner.textContent = [
      left === null ? "" : `Demo time remaining: ${remainingLabel(left)}. Save or submit before it ends.`,
      left !== null || recording ? "Public demo data only." : "",
    ].filter(Boolean).join(" ");
    if (performance.now() >= nextSync) void sync();
  }

  function track(kind, target) {
    const now = performance.now();
    if (!trackActivity || !active || !recording || inFlight >= 2 || now - lastEvent < 100) return;
    if (!/^[a-zA-Z][a-zA-Z0-9_.:-]{0,79}$/.test(target)) return;
    lastEvent = now;
    inFlight++;
    void fetch("/api/demo-access/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, target, surface }),
    }).catch(() => {}).finally(() => inFlight--);
  }

  function clicked(event) {
    const control = event.target.closest?.("button,a,select");
    if (!control || control.closest("#auth")) return;
    // Deliberately exclude textContent, URLs, form values and raw coordinates.
    track("click", control.dataset.action || control.dataset.page || control.id || "control");
  }
  function visible() {
    if (active && !document.hidden) { nextSync = 0; tick(); }
  }
  if (trackActivity) document.addEventListener("click", clicked, { capture: true });
  document.addEventListener("visibilitychange", visible);

  return {
    stop,
    dispose() {
      stop();
      document.removeEventListener("click", clicked, { capture: true });
      document.removeEventListener("visibilitychange", visible);
    },
    async start() {
      stop();
      active = true;
      const started = generation;
      await sync();
      if (!active || started !== generation) return;
      tick();
      if (active) timer = setInterval(tick, 1000);
      track("open", surface);
    },
  };
}
