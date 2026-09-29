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

import { bindClipboard } from "./desktop-clipboard.js";
import { bindDesktopVoice } from "./desktop-voice.js";

// Keep noVNC's touch gestures and mobile keyboard while binding it to this session.
const root = location.pathname.slice(0, location.pathname.lastIndexOf("/") + 1);
const { default: UI } = await import(`${root}app/ui.js`);
const identifier = location.pathname.split("/")[2];
const { pasteIntoViewer } = bindClipboard(UI);
bindDesktopVoice(UI, pasteIntoViewer);
for (const [callback, state] of [
  ["connectFinished", "connected"],
  ["disconnectFinished", "disconnected"],
]) {
  const handler = UI[callback];
  UI[callback] = (event) => {
    handler(event);
    window.parent.postMessage(
      { type: `monailabel-desktop-${state}` },
      location.origin,
    );
  };
}
await UI.start({
  settings: {
    defaults: { quality: 9, compression: 2, show_dot: true },
    mandatory: {
      // Resize the native desktop to this browser, including existing saved settings.
      resize: "remote",
      host: "",
      port: 0,
      encrypt: location.protocol === "https:",
      path: `/desktop/${identifier}/socket`,
      autoconnect: true,
      shared: true,
      reconnect: false,
    },
  },
});
