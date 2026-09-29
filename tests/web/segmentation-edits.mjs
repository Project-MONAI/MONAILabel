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
import { clearMask } from "../../packages/viewers/src/monailabel/viewers/resources/ohif/extension/src/segmentation-edits.js";

for (const axis of [0, 1, 2]) {
  test(`OHIF clearing preserves other labels and source slices on axis ${axis}`, () => {
    const shape = [4, 5, 6];
    const original = Uint8Array.from({ length: 120 }, (_, i) =>
      i % 3 ? 1 : 2,
    );
    const current = original.slice();
    const result = clearMask(
      current,
      { label_ids: [1], slice: { axis, index: 2 } },
      shape,
    );
    assert.deepEqual(current, original);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 5; j++)
        for (let k = 0; k < 6; k++) {
          const n = (i * 5 + j) * 6 + k;
          assert.equal(
            result[n],
            [i, j, k][axis] === 2 && original[n] === 1 ? 0 : original[n],
          );
        }
  });
}
test("invalid clear requests cannot change the source mask", () => {
  const current = Uint8Array.of(1, 2, 1, 2);
  for (const action of [
    { label_ids: [] },
    { label_ids: [0] },
    { label_ids: [1], slice: { axis: 3, index: 0 } },
    { label_ids: [1], slice: { axis: 0, index: 2 } },
    { label_ids: [1], image_region: {} },
  ])
    assert.throws(() => clearMask(current, action, [2, 1, 2]));
  assert.deepEqual(current, Uint8Array.of(1, 2, 1, 2));
  assert.deepEqual(
    clearMask(current, { label_ids: [1, 2] }, [2, 1, 2]),
    new Uint8Array(4),
  );
});
