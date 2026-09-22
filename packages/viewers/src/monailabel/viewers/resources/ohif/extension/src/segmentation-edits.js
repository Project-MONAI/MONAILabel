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
