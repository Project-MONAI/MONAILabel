/** Browser speech shared by the web workspace and OHIF. No provider key or audio storage. */
export function createSpeech({
  getText,
  onText,
  onState,
  onError,
  language = navigator.language || "en-US",
}) {
  const Recognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  const unavailable = !window.isSecureContext
    ? "Voice input needs HTTPS when connecting from another device."
    : !Recognition
      ? "This browser has no speech recognition. Use Safari/Chrome, or your keyboard's dictation."
      : "";
  let recognition = null;
  let disposed = false;
  const changeState = (state) => {
    if (!disposed) onState(state);
  };
  function cancel() {
    const current = recognition;
    recognition = null;
    if (current) current.abort();
    changeState("idle");
  }
  function start() {
    if (unavailable) {
      onError(unavailable);
      return;
    }
    cancel();
    window.speechSynthesis?.cancel();
    const prefix = getText().trim();
    const current = new Recognition();
    recognition = current;
    current.lang = language;
    current.continuous = false;
    current.interimResults = true;
    changeState("starting");
    current.onstart = () => {
      if (recognition === current) changeState("listening");
    };
    current.onresult = (event) => {
      if (recognition !== current || disposed) return;
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(" ")
        .trim();
      onText([prefix, transcript].filter(Boolean).join(" "));
    };
    current.onerror = (event) => {
      if (recognition !== current || disposed) return;
      const errors = {
        "not-allowed":
          "Microphone access was denied. Allow it in browser settings, then try again.",
        "service-not-allowed":
          "Speech recognition is unavailable. On iPad, enable Siri and use Safari.",
        "audio-capture":
          "No microphone is available. Check your device's microphone settings.",
        network:
          "The speech service could not connect. Check your connection or use keyboard dictation.",
        "no-speech": "No speech detected. Tap the microphone and try again.",
        "language-not-supported":
          "Your browser's speech service does not support this language.",
      };
      recognition = null;
      changeState("idle");
      if (event.error !== "aborted")
        onError(
          errors[event.error] ||
            "Speech input failed. You can edit or type the prompt.",
        );
    };
    current.onend = () => {
      if (recognition !== current || disposed) return;
      recognition = null;
      changeState("idle");
    };
    try {
      current.start();
    } catch {
      recognition = null;
      changeState("idle");
      onError(
        "Could not start dictation. Check microphone permission and try again.",
      );
    }
  }
  function speak(text) {
    if (disposed || !window.speechSynthesis || !window.SpeechSynthesisUtterance)
      return;
    cancel();
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language;
    window.speechSynthesis.speak(utterance);
  }
  return {
    available: !unavailable,
    unavailable,
    canSpeak: !!(window.speechSynthesis && window.SpeechSynthesisUtterance),
    start,
    stop() {
      recognition?.stop();
    },
    cancel,
    speak,
    stopSpeaking() {
      window.speechSynthesis?.cancel();
    },
    dispose() {
      cancel();
      disposed = true;
      window.speechSynthesis?.cancel();
    },
  };
}
