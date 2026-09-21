// Keep clipboard gestures in the viewer frame so HTTP LAN origins work too.
export function bindClipboard(UI) {
  const field = document.querySelector("#noVNC_clipboard_text");
  const panel = document.querySelector("#noVNC_clipboard");
  const note = document.createElement("p");
  note.setAttribute("role", "status");
  note.textContent = "Copy text in the viewer, or paste text here to send it.";
  let copyingUntil = 0;

  function copyToComputer(text) {
    const focused = document.activeElement;
    const temporary = document.createElement("textarea");
    temporary.value = text;
    temporary.style.position = "fixed";
    temporary.style.opacity = "0";
    document.body.append(temporary);
    temporary.select();
    let copied = false;
    try {
      // The user gesture permits this on HTTP, where navigator.clipboard is absent.
      copied = document.execCommand("copy");
    } catch {
      copied = false;
    } finally {
      temporary.remove();
      focused?.focus();
    }
    note.textContent = copied
      ? "Copied to your computer."
      : "Select the text above and copy it with your keyboard.";
    return copied;
  }

  function pasteIntoViewer(text) {
    if (!UI.rfb) return;
    UI.rfb.clipboardPasteFrom(text);
    UI.closeClipboardPanel();
    UI.rfb.focus();
    // Release a locally held modifier before sending the Linux viewer's paste gesture.
    UI.rfb.sendKey(0xffe3, "ControlLeft", false);
    UI.rfb.sendKey(0xffe7, "MetaLeft", false);
    UI.rfb.sendKey(0xffe3, "ControlLeft", true);
    UI.rfb.sendKey(0x76, "KeyV", true);
    UI.rfb.sendKey(0x76, "KeyV", false);
    UI.rfb.sendKey(0xffe3, "ControlLeft", false);
    note.textContent = "Pasted into the viewer.";
  }

  const receive = UI.clipboardReceive;
  UI.clipboardReceive = (event) => {
    receive(event);
    if (Date.now() < copyingUntil) {
      copyingUntil = 0;
      copyToComputer(event.detail.text);
    }
  };

  for (const [label, action] of [
    ["Copy to computer", () => copyToComputer(field.value)],
    ["Paste into viewer", () => pasteIntoViewer(field.value)],
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", action);
    panel.append(button);
  }
  panel.append(note);

  const editing = (target) =>
    target instanceof HTMLElement &&
    (target.matches("input, textarea") || target.isContentEditable);
  document.addEventListener(
    "keydown",
    (event) => {
      if (editing(event.target) || !(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLowerCase();
      if (key === "c" || key === "x") copyingUntil = Date.now() + 5000;
      if (key === "v") {
        // Let the browser provide clipboardData instead of forwarding a stale paste.
        event.stopImmediatePropagation();
      }
    },
    true,
  );
  document.addEventListener("paste", (event) => {
    if (editing(event.target)) return;
    const text = event.clipboardData?.getData("text/plain");
    if (text === undefined) return;
    event.preventDefault();
    pasteIntoViewer(text);
  });
}
