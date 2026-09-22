/* Scoped native edits. Plan everything before committing one undoable change. */
export async function clearAnnotations(job, request, labelIDs) {
  const { start, stop, client_id: clientID } = request;
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(stop) ||
    start < job.startFrame ||
    stop <= start ||
    stop > job.stopFrame + 1
  )
    throw new Error("The clear range is outside this video.");
  if (
    !labelIDs.length ||
    labelIDs.some((id) => !job.labels.some((label) => label.id === id))
  )
    throw new Error(
      "The requested tool label is unavailable in this CVAT task.",
    );

  const draft = await job.annotations.export();
  const before = JSON.stringify(draft);
  const appended = { tracks: [] };
  const removed = { tracks: [], shapes: [], tags: [] };
  const frames = new Map();
  const states = async (frame) => {
    if (!frames.has(frame))
      frames.set(frame, await job.annotations.get(frame, true, []));
    return frames.get(frame);
  };
  const selected = (object) =>
    labelIDs.includes(object.label_id) &&
    (clientID === null || object.clientID === clientID);
  const editable = async (object, frame) => {
    const state = (await states(frame)).find(
      (s) => s.clientID === object.clientID,
    );
    if (!state || state.lock)
      throw new Error(
        "A requested annotation is unavailable or locked. Unlock it and retry.",
      );
    return state;
  };
  const keyAt = async (track, frame) => {
    const existing = track.shapes.find((key) => key.frame === frame);
    if (existing) return { ...existing, points: [...existing.points] };
    const state = await editable(track, frame);
    const mutable = new Set(
      state.label.attributes.filter((a) => a.mutable).map((a) => a.id),
    );
    return {
      type: state.shapeType,
      frame,
      points: [...state.points],
      outside: state.outside,
      occluded: state.occluded,
      rotation: state.rotation,
      z_order: state.zOrder,
      attributes: Object.entries(state.attributes)
        .filter(([id]) => mutable.has(Number(id)))
        .map(([id, value]) => ({ spec_id: Number(id), value })),
    };
  };

  for (const track of draft.tracks) {
    if (!selected(track)) continue;
    const keys = [...track.shapes].sort((a, b) => a.frame - b.frame);
    if (
      !keys.some(
        (key, i) =>
          !key.outside &&
          key.frame < stop &&
          (keys[i + 1]?.frame ?? job.stopFrame + 1) > start,
      )
    )
      continue;
    await editable(track, Math.max(start, keys[0].frame));
    if (keys.some((key) => !["rectangle", "polygon"].includes(key.type)))
      throw new Error("Scoped clearing supports rectangle and polygon tracks.");
    removed.tracks.push(track);
    if (start === job.startFrame && stop === job.stopFrame + 1) continue;

    const replacements = new Map(keys.map((key) => [key.frame, key]));
    const first = Math.max(start, keys[0].frame);
    // Preserve the native interpolation on each side of the edited interval.
    // Densify only its two adjacent interpolation spans, including polygons
    // whose vertex correspondence changes between keyframes.
    const previous = keys.filter((key) => key.frame < first).at(-1);
    const next = keys.find((key) => key.frame >= stop);
    if (previous) {
      for (let frame = previous.frame + 1; frame < first; frame++)
        replacements.set(frame, await keyAt(track, frame));
    }
    if (stop <= job.stopFrame) {
      const end = next?.frame ?? stop;
      for (let frame = stop; frame <= end; frame++)
        replacements.set(frame, await keyAt(track, frame));
    }
    replacements.set(first, await keyAt(track, first));
    const shapes = [...replacements.values()]
      .sort((a, b) => a.frame - b.frame)
      .map((key) =>
        key.frame >= start && key.frame < stop
          ? { ...key, outside: true }
          : key,
      );
    // Retain the server ID: saving updates this track instead of deleting its
    // submitted identity. CVAT assigns a fresh local ID to the replacement.
    const { clientID: _localID, ...replacement } = track;
    appended.tracks.push({ ...replacement, shapes });
  }
  for (const kind of ["shapes", "tags"]) {
    for (const object of draft[kind] || []) {
      if (selected(object) && object.frame >= start && object.frame < stop) {
        await editable(object, object.frame);
        removed[kind].push(object);
      }
    }
  }
  if (JSON.stringify(await job.annotations.export()) !== before)
    throw new Error(
      "The CVAT draft changed while preparing the edit. Retry to keep your changes.",
    );
  const count = Object.values(removed).reduce(
    (sum, objects) => sum + objects.length,
    0,
  );
  if (count) await job.annotations.commit(appended, removed, start);
  return count;
}
