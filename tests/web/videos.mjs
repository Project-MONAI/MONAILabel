import assert from "node:assert/strict";
import test from "node:test";
import {
  videoButtons,
  videoReviews,
} from "../../packages/server/src/monailabel/server/static/videos.js";

const state = {
  page: "datasets",
  searches: { datasets: "", review: "" },
  datasetFilter: "all",
  reviewFilter: "pending",
  roles: ["annotator"],
  videoCapabilities: { cvat: true },
  videos: [
    {
      id: "clip",
      name: '<img src=x onerror="alert(1)">',
      group_id: "procedure",
      width: 64,
      height: 48,
      frames: 6,
      duration: 0.8,
      revision: 0,
      split: "pool",
    },
  ],
};

test("video actions respect capability and roles", () => {
  const html = videoButtons(state, state.videos[0]);
  assert.ok(html.includes("CVAT"));
  assert.ok(!html.includes("<img"));
  assert.ok(!html.includes("Delete clip"));
  assert.ok(
    !videoButtons(
      { ...state, videoCapabilities: {} },
      state.videos[0],
    ).includes('data-action="video-open"'),
  );
  assert.equal(videoButtons({ ...state, roles: [] }, state.videos[0]), "");
});

test("video reviews show only submitted revisions matching the active filter", () => {
  const review = { ...state, page: "review", roles: ["reviewer"] };
  assert.ok(!videoReviews(review, () => "pending").includes("Inspect in CVAT"));
  review.videos = [
    { ...state.videos[0], annotation_id: "revision", revision: 1 },
  ];
  const html = videoReviews(review, () => "pending");
  assert.ok(html.includes("Inspect in CVAT"));
  assert.ok(html.includes("Review revision"));
  assert.ok(!html.includes('data-action="video-open"'));
  assert.ok(
    !videoReviews(review, () => "accepted").includes("Review revision"),
  );
});
