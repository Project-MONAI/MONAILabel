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
import { demoSession, remainingLabel } from "../../packages/server/src/monailabel/server/static/demo-session.js";

test("remaining time is readable and never negative", () => {
  assert.equal(remainingLabel(3600), "01:00:00");
  assert.equal(remainingLabel(61.1), "00:01:02");
  assert.equal(remainingLabel(-1), "00:00:00");
});

function environment(t, info) {
  const document = Object.assign(new EventTarget(), { hidden: false });
  const previous = globalThis.document;
  globalThis.document = document;
  t.after(() => {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  });
  let tick, now = 1000;
  t.mock.method(performance, "now", () => now);
  t.mock.method(globalThis, "setInterval", (callback) => { tick = callback; return 1; });
  t.mock.method(globalThis, "clearInterval", () => { tick = null; });
  const fetch = t.mock.method(globalThis, "fetch", async () => ({ status: 200, ok: true, json: async () => info }));
  return { document, fetch, advance: (ms) => { now += ms; tick?.(); } };
}

const timed = {
  activity_recording: true,
  server_now: "2026-01-01T12:00:00Z",
  expires_at: "2026-01-01T12:01:00Z",
};

test("OHIF can reuse the deadline without adding telemetry; disposal removes listeners", async (t) => {
  const env = environment(t, timed);
  const banner = {};
  let expired = 0;
  const clock = demoSession({ banner, trackActivity: false, onExpired: () => expired++ });
  await clock.start();
  assert.equal(banner.hidden, false);
  assert.match(banner.textContent, /00:01:00/);
  assert.match(banner.textContent, /Public demo data only/);
  assert.doesNotMatch(banner.textContent, /recorded/);
  assert.equal(env.fetch.mock.callCount(), 1);
  env.advance(61000);
  assert.equal(expired, 1);
  assert.equal(banner.textContent, "Demo access has ended.");
  clock.dispose();
  env.document.dispatchEvent(new Event("visibilitychange"));
  env.document.dispatchEvent(new Event("click"));
  assert.equal(env.fetch.mock.callCount(), 1);
});

test("network failure does not extend a known deadline", async (t) => {
  const env = environment(t, timed);
  const banner = {};
  let expired = false;
  const clock = demoSession({ banner, trackActivity: false, onExpired: () => { expired = true; } });
  await clock.start();
  env.fetch.mock.mockImplementation(async () => { throw new Error("offline"); });
  env.advance(31000);
  await new Promise(setImmediate);
  assert.match(banner.textContent, /00:00:29/);
  env.advance(30000);
  assert.equal(expired, true);
  clock.dispose();
});

test("ordinary untimed installations do not gain a demo countdown", async (t) => {
  environment(t, { ...timed, activity_recording: false, expires_at: null });
  const banner = {};
  const clock = demoSession({ banner, trackActivity: false, onExpired: () => assert.fail("Unexpected expiry") });
  await clock.start();
  assert.equal(banner.hidden, true);
  assert.equal(banner.textContent, "");
  clock.dispose();
});
