import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
const source = readFileSync(
  new URL(
    "../../packages/viewers/src/monailabel/viewers/resources/ohif/extension/src/initialize.js",
    import.meta.url,
  ),
  "utf8",
);
const { initializeAnnotation } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

function fixture(annotation_id = "saved") {
  let imageReady = false;
  const calls = { loads: 0, prepares: 0, writes: 0 };
  const state = {
    asset: { annotation_id },
    transfer: {
      canPrepare: () => imageReady,
      prepare: async () => {
        calls.prepares++;
      },
      write: () => {
        calls.writes++;
      },
    },
  };
  const request = async () => {
    calls.loads++;
    return new Uint8Array([0, 1, 0]);
  };
  return {
    state,
    request,
    calls,
    ready: () => {
      imageReady = true;
    },
  };
}

test("waits for geometry and shares concurrent saved-mask loads", async () => {
  const f = fixture();
  assert.equal(await initializeAnnotation(f.state, f.request), false);
  assert.equal(f.calls.loads, 0);
  f.ready();
  await Promise.all([
    initializeAnnotation(f.state, f.request),
    initializeAnnotation(f.state, f.request),
  ]);
  assert.deepEqual(f.calls, { loads: 1, prepares: 1, writes: 1 });
  assert.equal(f.state.initialized, true);
  // Remount/layout change attaches the existing layer and never reloads over local edits.
  await initializeAnnotation(f.state, f.request);
  assert.deepEqual(f.calls, { loads: 1, prepares: 2, writes: 1 });
});
test("new samples create a layer without fetching or writing a saved mask", async () => {
  const f = fixture(null);
  f.ready();
  assert.equal(await initializeAnnotation(f.state, f.request), true);
  assert.deepEqual(f.calls, { loads: 0, prepares: 1, writes: 0 });
});
test("failed saved-mask request leaves the sample uninitialized and can retry", async () => {
  const f = fixture();
  f.ready();
  await assert.rejects(
    initializeAnnotation(f.state, async () => {
      throw new Error("Offline");
    }),
    /Offline/,
  );
  assert.equal(f.state.initialized, undefined);
  assert.equal(f.state.initializing, null);
  assert.equal(f.calls.prepares, 0);
  await initializeAnnotation(f.state, f.request);
  assert.equal(f.state.initialized, true);
  assert.equal(f.calls.writes, 1);
});
test("failed attachment is retryable and does not mark the mask restored", async () => {
  const f = fixture();
  f.ready();
  f.state.transfer.prepare = async () => {
    throw new Error("Viewport unavailable");
  };
  await assert.rejects(
    initializeAnnotation(f.state, f.request),
    /Viewport unavailable/,
  );
  assert.equal(f.state.initialized, undefined);
  assert.equal(f.calls.writes, 0);
  f.state.transfer.prepare = async () => {};
  await initializeAnnotation(f.state, f.request);
  assert.equal(f.calls.writes, 1);
});

test("leaving a viewer during a saved-mask fetch cannot initialize a different scene", async () => {
  const f = fixture();
  f.ready();
  let finish;
  const result = initializeAnnotation(
    f.state,
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  f.state.closed = true;
  finish(new Uint8Array([0, 1, 0]));
  assert.equal(await result, false);
  assert.equal(f.calls.prepares, 0);
  assert.equal(f.calls.writes, 0);
  assert.equal(await initializeAnnotation(f.state, f.request), false);
});
