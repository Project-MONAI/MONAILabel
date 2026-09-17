import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const source = readFileSync(new URL("../../packages/viewers/src/monailabel/viewers/resources/ohif/extension/src/speech.js", import.meta.url), "utf8");
const { createSpeech } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const instances = [];
class Recognition {
  constructor() { instances.push(this); }
  start() { this.onstart(); }
  stop() { this.onend(); }
  abort() { this.onend(); }
  emit(text) { this.onresult({ results: [[{ transcript: text }]] }); }
}
let spoken = [];
globalThis.window = { isSecureContext: true, webkitSpeechRecognition: Recognition, speechSynthesis: { cancel() {}, speak(item) { spoken.push(item.text); } }, SpeechSynthesisUtterance: class {} };
globalThis.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
let text = "Annotate", state, error;
const speech = createSpeech({ getText: () => text, onText: (value) => text = value, onState: (value) => state = value, onError: (value) => error = value, language: "en-US" });
assert(speech.available);
speech.start();
const first = instances.at(-1);
assert.equal(state, "listening");
first.emit("the spleen");
assert.equal(text, "Annotate the spleen");
first.emit("the spleen on this slice");
assert.equal(text, "Annotate the spleen on this slice");
speech.stop();
assert.equal(state, "idle");
speech.start();
const late = instances.at(-1);
speech.cancel();
late.emit("must not arrive after Send or a project switch");
assert.equal(text, "Annotate the spleen on this slice");
speech.start();
instances.at(-1).onerror({ error: "not-allowed" });
assert.match(error, /denied/);
assert.equal(state, "idle");
speech.speak("Annotation ready.");
assert.deepEqual(spoken, ["Annotation ready."]);
speech.dispose();
speech.speak("Must not speak after leaving.");
assert.equal(spoken.length, 1);
window.isSecureContext = false;
const insecure = createSpeech({ getText: () => "", onText() {}, onState() {}, onError: (value) => error = value, language: "en-US" });
assert.equal(insecure.available, false);
insecure.start();
assert.match(error, /HTTPS/);
window.isSecureContext = true;
delete window.webkitSpeechRecognition;
assert.match(createSpeech({ language: "en-US" }).unavailable, /keyboard/);
console.log("Speech lifecycle, transcript, permission, cancellation and TTS checks passed.");
