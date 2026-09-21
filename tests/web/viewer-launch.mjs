import assert from "node:assert/strict";
import test from "node:test";
import {
  closePendingViewer,
  openPreparedViewer,
  desktopTarget,
} from "../../packages/server/src/monailabel/server/static/viewer-launch.js";

globalThis.location = { origin: "http://workspace.test" };

test("desktop launch is native for loopback and streamed for remote addresses", () => {
  for (const host of ["localhost", "127.0.0.1", "127.0.0.2", "[::1]"])
    assert.equal(desktopTarget(new URL(`http://${host}:8000`)), "auto");
  for (const host of ["192.168.1.4", "workspace.example", "localhost.example"])
    assert.equal(desktopTarget(new URL(`https://${host}`)), "browser");
  assert.equal(
    desktopTarget(new URL("http://localhost:8000/?desktop=browser")),
    "browser",
  );
});

test("failed setup closes only its loading page and preserves other viewer drafts", () => {
  const closed = [];
  for (const href of [
    "about:blank",
    "http://workspace.test/static/viewer-launch.html",
    "http://workspace.test/cvat/editor/draft",
  ]) {
    closePendingViewer({ location: { href }, close: () => closed.push(href) });
  }
  closePendingViewer({
    get location() {
      throw new Error("Cross-origin viewer");
    },
    close: () => closed.push("external"),
  });
  assert.deepEqual(closed, [
    "about:blank",
    "http://workspace.test/static/viewer-launch.html",
  ]);
});

test("closing a loading tab never navigates another tab", () => {
  const tab = {
    closed: true,
    location: { replace: () => assert.fail("Closed tab reused") },
  };
  assert.equal(
    openPreparedViewer("http://workspace.test/cvat/editor/task", tab, true),
    false,
  );
});

test("a loading tab navigated to an existing draft is preserved on success", () => {
  const tab = {
    closed: false,
    location: {
      href: "http://workspace.test/cvat/editor/draft",
      replace: () => assert.fail("Draft replaced"),
    },
  };
  assert.equal(
    openPreparedViewer("http://workspace.test/cvat/editor/new", tab, true),
    false,
  );
});

test("prepared OHIF and CVAT viewers navigate an open reserved tab", () => {
  const navigated = [];
  const tab = {
    closed: false,
    location: {
      href: "http://workspace.test/static/viewer-launch.html",
      replace: (url) => navigated.push(url),
    },
  };
  assert.equal(
    openPreparedViewer("http://workspace.test/ohif/viewer", tab),
    true,
  );
  assert.equal(
    openPreparedViewer("http://workspace.test/cvat/editor/task", tab, true),
    true,
  );
  assert.equal(navigated.length, 2);
});
