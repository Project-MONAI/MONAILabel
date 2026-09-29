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

import { createSpeech } from "/static/speech.js";

// Dictation runs on the browser device, not the remote viewer's Linux desktop.
export function bindDesktopVoice(UI, pasteIntoViewer) {
  const style = document.createElement("link");
  style.rel = "stylesheet";
  style.href = "/static/desktop-voice.css";
  document.head.append(style);
  const button = document.createElement("button");
  button.id = "monailabel-voice-button";
  button.type = "button";
  button.className = "noVNC_button";
  button.title = "Dictate a prompt";
  button.setAttribute("aria-label", "Dictate a prompt");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", "monailabel-voice");
  button.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></svg>`;
  const wrapper = document.createElement("div");
  wrapper.className = "noVNC_vcenter";
  wrapper.innerHTML = `<section id="monailabel-voice" class="noVNC_panel" aria-label="Voice input">
    <div class="noVNC_heading">Dictate a prompt</div>
    <p>Click the viewer's prompt box first. Dictate, review the text, then insert it.</p>
    <textarea rows="4" aria-label="Dictated prompt" placeholder="Your words appear here…"></textarea>
    <div class="button_row">
      <button type="button" data-action="record" aria-pressed="false">Use microphone</button>
      <button type="button" data-action="insert">Insert into viewer</button>
    </div>
    <p role="status" aria-live="polite"></p>
  </section>`;
  document.querySelector("#noVNC_clipboard_button").before(button, wrapper);
  const panel = wrapper.firstElementChild;
  const field = panel.querySelector("textarea");
  const record = panel.querySelector('[data-action="record"]');
  const insert = panel.querySelector('[data-action="insert"]');
  const note = panel.querySelector('[role="status"]');
  let listening = false;
  function updateInsert() {
    insert.disabled =
      listening || !field.value.trim() || !UI.connected || UI.rfb?.viewOnly;
  }
  const speech = createSpeech({
    getText: () => field.value,
    onText: (text) => {
      field.value = text;
      updateInsert();
    },
    onState: (state) => {
      listening = state !== "idle";
      record.textContent = listening ? "Stop microphone" : "Use microphone";
      record.setAttribute("aria-pressed", String(listening));
      field.readOnly = listening;
      note.textContent =
        state === "starting"
          ? "Starting microphone…"
          : listening
            ? "Listening…"
            : "Review the text before inserting.";
      updateInsert();
    },
    onError: (text) => {
      note.textContent = text;
    },
  });
  record.disabled = !speech.available;
  record.title = speech.unavailable || "Use your browser's speech service";
  field.addEventListener("input", updateInsert);
  record.addEventListener("click", () =>
    listening ? speech.stop() : speech.start(),
  );
  function close() {
    speech.cancel();
    panel.classList.remove("noVNC_open");
    button.classList.remove("noVNC_selected");
    button.setAttribute("aria-expanded", "false");
  }
  const closeAll = UI.closeAllPanels;
  UI.closeAllPanels = () => {
    close();
    closeAll();
  };
  button.addEventListener("click", () => {
    if (panel.classList.contains("noVNC_open")) {
      close();
      return;
    }
    UI.closeAllPanels();
    UI.openControlbar();
    panel.classList.add("noVNC_open");
    button.classList.add("noVNC_selected");
    button.setAttribute("aria-expanded", "true");
    note.textContent =
      speech.unavailable || "Use microphone to start dictation.";
    updateInsert();
  });
  insert.addEventListener("click", () => {
    if (listening || !field.value.trim()) return;
    speech.cancel();
    if (!pasteIntoViewer(field.value)) {
      note.textContent =
        "Reconnect to the viewer before inserting. Your text is kept here.";
      return;
    }
    field.value = "";
    close();
  });
  const disconnected = UI.disconnectFinished;
  UI.disconnectFinished = (event) => {
    close();
    disconnected(event);
  };
  window.addEventListener("pagehide", () => speech.dispose());
}
