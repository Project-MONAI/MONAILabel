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
import test from "node:test";
import { recordedTrainingCardsHTML, recordedTrainingHTML } from "../../packages/server/src/monailabel/server/static/demo-training.js";

test("recorded examples are labelled, escaped, and never invent missing scores", () => {
  const example = {
    name: "<script>bad()</script>", dataset: "Public test fixture", completed_at: "2026-01-01T00:00:00Z",
    training_cases: 3, evaluation_cases: 0, epochs: 1, steps_per_epoch: 2,
    duration_seconds: 1.5, checkpoint_sha256: "a".repeat(64),
    mean_dice: null, final_loss: null,
    caveats: ["<img onerror=bad()>"], log: "</textarea><script>bad()</script>",
  };
  const html = recordedTrainingHTML(example);
  assert.match(html, /Recorded run — completed before your visit/);
  assert.match(html, /Your current corrections were not used/);
  assert.match(html, /No held-out score is available/);
  assert.doesNotMatch(html, /<script>|<img|final training loss|progress/);
  assert.match(html, /&lt;script&gt;/);
  const measured = recordedTrainingHTML({ ...example, mean_dice: 0, final_loss: 0 });
  assert.match(measured, /mean Dice: 0.000/);
  assert.match(measured, /final training loss: 0.0000/);
});

test("both recorded examples have visible cards, not a modal chooser", () => {
  const html = recordedTrainingCardsHTML();
  assert.match(html, /nnU-Net training/);
  assert.match(html, /VISTA3D fine-tuning/);
  assert.match(html, /no training will run/);
  assert.equal((html.match(/data-recorded-recipe=/g) || []).length, 2);
  assert.doesNotMatch(html, /<dialog|<select|data-action="start-training"/);
});
