import { createSpeech } from "/static/speech.js";

export function voiceInput(input, button, status, readReplies) {
  let listening = false;
  const speech = createSpeech({
    getText: () => input.value,
    onText: (text) => {
      input.value = text;
    },
    onState: (state) => {
      listening = state !== "idle";
      button.setAttribute(
        "aria-label",
        listening ? "Stop microphone" : "Use microphone",
      );
      button.title = listening
        ? "Stop microphone"
        : speech.unavailable || "Use microphone";
      button.setAttribute("aria-pressed", String(listening));
      button.classList.toggle("listening", listening);
      status.textContent =
        state === "starting"
          ? "Starting microphone…"
          : state === "listening"
            ? "Listening… tap the stop button when finished."
            : speech.unavailable;
    },
    onError: (text) => {
      status.textContent = text;
    },
  });
  button.disabled = !speech.available;
  button.title =
    speech.unavailable ||
    "Speak using your browser's speech service. Review the transcript before sending.";
  readReplies.disabled = !speech.canSpeak;
  if (!speech.available) status.textContent = speech.unavailable;
  button.addEventListener("click", () =>
    listening ? speech.stop() : speech.start(),
  );
  readReplies.addEventListener("change", () => {
    if (readReplies.checked) speech.speak("Spoken replies on.");
    else speech.stopSpeaking();
  });
  window.addEventListener("pagehide", () => speech.dispose());
  return {
    cancel: speech.cancel,
    speak(text) {
      if (readReplies.checked) speech.speak(text);
    },
  };
}
