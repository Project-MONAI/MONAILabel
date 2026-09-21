import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(
  new URL(
    "../../packages/server/src/monailabel/server/static/desktop-lifecycle.js",
    import.meta.url,
  ),
  "utf8",
);
const settle = () => new Promise(setImmediate);

function desktop(fetch) {
  const handlers = new Map();
  const timers = new Map();
  const result = { closed: 0, returned: [] };
  const frame = { contentWindow: {} };
  const origin = "http://workspace.test";
  const watch = runInNewContext(
    source.replace("export function", "function") + "; watchDesktop;",
    {
      fetch,
      AbortSignal,
      window: {
        addEventListener: (name, callback) => handlers.set(name, callback),
        close: () => result.closed++,
      },
      location: { origin, replace: (url) => result.returned.push(url) },
      setTimeout: (callback, delay) => {
        const id = {};
        timers.set(id, { callback, delay });
        return id;
      },
      clearTimeout: (id) => timers.delete(id),
    },
  );
  watch(frame, "viewer", "/datasets?project=project");
  return {
    result,
    message: (type, overrides = {}) =>
      handlers.get("message")({
        origin,
        source: frame.contentWindow,
        data: { type: `monailabel-desktop-${type}` },
        ...overrides,
      }),
    leave: () => handlers.get("pagehide")(),
    async tick(delay) {
      for (const [id, timer] of [...timers]) {
        if (timer.delay !== delay) continue;
        timers.delete(id);
        timer.callback();
      }
      await settle();
    },
  };
}

test("only a confirmed native exit closes the tab, with a workspace fallback", async () => {
  const replies = [200, 503, new Error("Offline"), 410];
  const view = desktop(async () => {
    const status = replies.shift();
    if (status instanceof Error) throw status;
    return { status };
  });
  view.message("disconnected");
  await settle();
  assert.equal(view.result.closed, 0);
  await view.tick(2000);
  assert.equal(view.result.closed, 0);
  await view.tick(2000);
  assert.equal(view.result.closed, 0);
  await view.tick(2000);
  assert.equal(view.result.closed, 1);
  await view.tick(100);
  assert.deepEqual(view.result.returned, ["/datasets?project=project"]);
});

for (const action of ["reconnect", "navigate"])
  test(`${action} ignores an obsolete exit response`, async () => {
    let respond;
    let requests = 0;
    const view = desktop(() => {
      requests++;
      return new Promise((resolve) => {
        respond = resolve;
      });
    });
    view.message("disconnected");
    view.message("disconnected");
    assert.equal(requests, 1);
    if (action === "reconnect") view.message("connected");
    else view.leave();
    respond({ status: 410 });
    await settle();
    await view.tick(2000);
    assert.equal(requests, 1);
    assert.equal(view.result.closed, 0);
  });

test("foreign messages and lost access cannot close another viewer", async () => {
  let requests = 0;
  const view = desktop(async () => {
    requests++;
    return { status: 403 };
  });
  view.message("disconnected", { origin: "http://other.test" });
  view.message("disconnected", { source: {} });
  assert.equal(requests, 0);
  view.message("disconnected");
  await settle();
  await view.tick(2000);
  assert.equal(requests, 1);
  assert.equal(view.result.closed, 0);
});
