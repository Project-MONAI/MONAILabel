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

/** Source-grid edits leave the input snapshot intact for undo. */
export function clearMask(current, action, shape) {
  const slice = action.slice;
  if (
    action.image_region ||
    shape.length !== 3 ||
    shape.some((n) => !Number.isInteger(n) || n < 1) ||
    current.length !== shape.reduce((a, b) => a * b, 1)
  )
    throw new Error("Clear request does not match the source volume.");
  if (
    slice &&
    (!Number.isInteger(slice.axis) ||
      slice.axis < 0 ||
      slice.axis > 2 ||
      !Number.isInteger(slice.index) ||
      slice.index < 0 ||
      slice.index >= shape[slice.axis])
  )
    throw new Error("Choose a valid source slice to clear.");
  const labels = new Set(action.label_ids);
  if (
    !labels.size ||
    [...labels].some((id) => !Number.isInteger(id) || id < 1 || id > 255)
  )
    throw new Error("Choose foreground labels to clear.");
  const result = current.slice();
  const stride = slice
    ? shape.slice(slice.axis + 1).reduce((a, b) => a * b, 1)
    : 1;
  for (let i = 0; i < result.length; i++) {
    if (slice && Math.floor(i / stride) % shape[slice.axis] !== slice.index)
      continue;
    if (labels.has(result[i])) result[i] = 0;
  }
  return result;
}
