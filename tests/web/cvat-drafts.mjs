import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(
  new URL("../../packages/server/src/monailabel/server/static/cvat-bridge.js", import.meta.url),
  "utf8",
);
function fixture() {
  const draft = { tracks: [{ shapes: [{ points: [1, 2, 3, 4] }] }] };
  const committed = [];
  const instance = {
    stopFrame: 3,
    labels: [{ id: 1 }],
    annotations: {
      export: async () => structuredClone(draft),
      commit: async (change) => {
        committed.push(change);
        draft.tracks.push(...change.tracks);
      },
    },
  };
  let ready;
  const window = {
    cvatUI: {
      registerComponent: (register) => register({
        store: {
          subscribe: () => () => {},
          getState: () => ({ annotation: {
            job: { instance, fetching: false },
            player: { frame: { number: 0, data: {}, fetching: false } },
            annotations: { activatedStateID: null, states: [] },
          } }),
        },
        dispatch: async () => {},
        actionCreators: { changeFrameAsync: () => ({}) },
      }),
    },
  };
  runInNewContext(source, {
    window,
    document: { addEventListener: (_, callback) => { ready = callback; } },
    // Match the APIs available on a non-local HTTP origin: no crypto.subtle.
    crypto: { getRandomValues: (bytes) => webcrypto.getRandomValues(bytes) },
  });
  ready();
  return { bridge: window.monaiVideo, draft, committed };
}
function proposal(signature) {
  return {
    request: { draft_signature: signature, client_id: null, seed: { frame: 0 } },
    keyframes: [{ frame: 0, box: [5, 6, 7, 8], outside: false, occluded: false }],
  };
}
test("unchanged CVAT drafts retain their token and accept results over HTTP", async () => {
  const { bridge, committed } = fixture();
  const context = await bridge.context();
  assert.match(context.draft_signature, /^[a-f0-9]{64}$/);
  assert.equal(await bridge.snapshot(), context.draft_signature);
  await bridge.apply(proposal(context.draft_signature), 1);
  assert.equal(committed.length, 1);
  assert.notEqual(await bridge.snapshot(), context.draft_signature);
});
test("editing an unrelated keyframe prevents a proposal from changing the draft", async () => {
  const { bridge, draft, committed } = fixture();
  const signature = await bridge.snapshot();
  draft.tracks[0].shapes[0].points[2] = 99;
  await assert.rejects(bridge.apply(proposal(signature), 1), /draft changed/);
  assert.equal(committed.length, 0);
  assert.equal(draft.tracks[0].shapes[0].points[2], 99);
});
test("a reloaded native session cannot accept a previous session's proposal", async () => {
  const old = fixture();
  const fresh = fixture();
  await assert.rejects(fresh.bridge.apply(proposal(await old.bridge.snapshot()), 1), /draft changed/);
  assert.equal(fresh.committed.length, 0);
});
