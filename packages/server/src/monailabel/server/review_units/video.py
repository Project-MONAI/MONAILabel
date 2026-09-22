"""Detect affected time intervals, including changes to interpolated tracks."""

from bisect import bisect_right

import numpy as np

from monailabel.core.video import ObjectTrack, TrackDocument


def state(track: ObjectTrack | None, frames: list[int], frame: int) -> object:
    position = bisect_right(frames, frame) - 1
    if track is None or position < 0:
        return None
    key = track.keyframes[position]
    if key.outside:
        return None
    following = track.keyframes[position + 1] if position + 1 < len(track.keyframes) else None
    # Exact keyframes do not depend on the next interpolation endpoint.
    return track.label_id, key, following if key.frame != frame else None


def changed_frames(before: TrackDocument, after: TrackDocument, count: int) -> set[int]:
    old = {track.id: track for track in before.tracks}
    new = {track.id: track for track in after.tracks}
    changed = np.zeros(count, dtype=bool)
    for identifier in old.keys() | new.keys():
        first, second = old.get(identifier), new.get(identifier)
        if first == second:
            continue
        tracks = [first, second]
        frames = [[key.frame for key in track.keyframes] if track else [] for track in tracks]

        boundaries = sorted({0, count, *frames[0], *frames[1]})
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            if state(first, frames[0], start) != state(second, frames[1], start):
                changed[start] = True
            if stop > start + 1 and state(first, frames[0], start + 1) != state(
                second, frames[1], start + 1
            ):
                changed[start + 1 : stop] = True
    return set(np.flatnonzero(changed).tolist())
