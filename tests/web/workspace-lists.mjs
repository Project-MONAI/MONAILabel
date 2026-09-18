import assert from "node:assert/strict";
import test from "node:test";
import { datasets } from "../../packages/server/src/monailabel/server/static/workspace-lists.js";

test("dataset filters render evaluation names as text", () => {
  const html = datasets(
    {
      assets: [],
      evaluationSets: [
        { id: "fixed", name: '</option><img src=x onerror="alert(1)">' },
      ],
      searches: { datasets: "" },
      datasetFilter: "all",
      pages: { datasets: 1 },
      selectedFiles: new Set(),
    },
    false,
    () => "pending",
  );
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;/option&gt;&lt;img"));
});

import {
  pageSlice,
  visibleAssets,
} from "../../packages/server/src/monailabel/server/static/workspace-lists.js";
import { evaluationSetAction } from "../../packages/server/src/monailabel/server/static/evaluation-sets.js";

function mixedState() {
  const images = Array.from({ length: 11 }, (_, i) => ({
    id: `image-${i}`,
    name: `Image ${i}`,
    group_id: `patient-${i}`,
    kind: i ? "image2d" : "volume3d",
    spatial_shape: [32, 24],
    revision: 0,
    split: "pool",
    created_at: String(i * 2).padStart(2, "0"),
  }));
  const video = {
    id: "clip",
    name: '<img src=x onerror="alert(1)">',
    group_id: "procedure-a",
    kind: "video",
    width: 64,
    height: 48,
    frames: 6,
    duration: 0.8,
    revision: 0,
    split: "pool",
    created_at: "05",
  };
  return {
    assets: images,
    videos: [video],
    evaluationSets: [],
    searches: { datasets: "" },
    datasetFilter: "all",
    pages: { datasets: 1 },
    selectedFiles: new Set([video.id]),
    context: {},
    roles: ["manager"],
    videoCapabilities: { cvat: true },
  };
}

test("mixed samples share one paginated dataset table and selection", () => {
  const state = mixedState();
  const html = datasets(state, true, () => "unannotated");
  assert.equal((html.match(/<table /g) || []).length, 1);
  assert.equal((html.match(/data-file-selection=/g) || []).length, 10);
  assert.ok(html.includes('data-file-selection="clip"'));
  assert.ok(
    html.includes(
      'Select &lt;img src=x onerror=&quot;alert(1)&quot;&gt;" checked',
    ),
  );
  assert.ok(html.includes('data-action="video-open"'));
  assert.ok(html.includes('data-action="ohif"'));
  assert.ok(html.includes('data-action="viewer"'));
  assert.ok(html.includes("1–10 of 12"));
  assert.ok(!html.includes("<img") && !html.includes("Video clips"));
  assert.equal(visibleAssets(state, () => "unannotated")[3].id, "clip");
  state.pages.datasets = 2;
  assert.equal(
    pageSlice(
      visibleAssets(state, () => "unannotated"),
      state,
      "datasets",
    ).length,
    2,
  );
  assert.ok(datasets(state, true, () => "unannotated").includes("11–12 of 12"));
});

test("video-only filters, grouping, empty results and viewer permissions use the shared list", () => {
  const state = mixedState();
  state.searches.datasets = "  PROCEDURE-A  ";
  assert.deepEqual(
    visibleAssets(state, () => "unannotated").map((a) => a.id),
    ["clip"],
  );
  state.searches.datasets = "";
  state.datasetFilter = "kind:video";
  assert.deepEqual(
    visibleAssets(state, () => "unannotated").map((a) => a.id),
    ["clip"],
  );
  state.datasetFilter = "set:held-out";
  state.evaluationSets = [
    { id: "held-out", name: "Held out", member_groups: ["procedure-a"] },
  ];
  assert.deepEqual(
    visibleAssets(state, () => "unannotated").map((a) => a.id),
    ["clip"],
  );
  state.datasetFilter = "all";
  state.assets = [];
  assert.ok(datasets(state, true, () => "unannotated").includes("1–1 of 1"));
  state.searches.datasets = "missing";
  const empty = datasets(state, true, () => "unannotated");
  assert.ok(empty.includes("No samples match") && empty.includes("0 results"));
  state.searches.datasets = "";
  state.roles = [];
  const reader = datasets(state, false, () => "unannotated");
  assert.ok(
    !reader.includes("data-file-selection") &&
      !reader.includes('data-action="video-open"'),
  );
});

test("video selections cannot be sent to fixed image evaluation endpoints", async () => {
  const state = mixedState();
  state.project = { id: "project" };
  let called = false;
  await assert.rejects(
    evaluationSetAction("evaluation-set-create", null, {
      state,
      api: () => {
        called = true;
      },
    }),
    /video tracking evaluation is not available/,
  );
  assert.equal(called, false);
});
