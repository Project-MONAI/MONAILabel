/** Restore once per sample session; duplicate viewport events share the same work.
 * Later viewport/layout changes attach the existing layer without replacing edits.
 */
export async function initializeAnnotation(state, request) {
  if (state.closed || !state.transfer?.canPrepare()) return false;
  if (!state.initializing) {
    state.initializing = (async () => {
      const mask =
        !state.initialized && state.asset.annotation_id
          ? await request(
              "/annotations/" + state.asset.annotation_id + "/mask.bin",
              undefined,
              true,
            )
          : null;
      if (state.closed) return false;
      await state.transfer.prepare();
      if (state.closed) return false;
      if (mask) state.transfer.write(mask);
      state.initialized = true;
      return true;
    })().finally(() => {
      state.initializing = null;
    });
  }
  return state.initializing;
}
