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
