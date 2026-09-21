import { bindClipboard } from "./desktop-clipboard.js";

// Keep noVNC's touch gestures and mobile keyboard while binding it to this session.
const root = location.pathname.slice(0, location.pathname.lastIndexOf("/") + 1);
const { default: UI } = await import(`${root}app/ui.js`);
const identifier = location.pathname.split("/")[2];
bindClipboard(UI);
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
