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
instances.at(-1).onend();
assert.match(error, /No speech was transcribed/);
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
const count = instances.length;
speech.start();
assert.equal(instances.length, count);
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
