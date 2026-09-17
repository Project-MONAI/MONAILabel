"""Voxel-grid operations shared by the standalone Slicer bridge and its tests."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def slicer_reverses_slices(
    source_affine: NDArray[Any], loaded_affine: NDArray[Any], shape: list[int]
) -> bool:
    """Recognize Slicer's lossless K reversal when loading a left-handed grid.

    Keep Slicer's right-handed display grid; masks and prompts use source indices.
    Other geometry changes cannot be mapped by reversing slices and must fail.
    """
    if np.allclose(loaded_affine, source_affine, rtol=0, atol=1e-4):
        return False
    reversal = np.eye(4)
    reversal[2, 2], reversal[2, 3] = -1, shape[2] - 1
    if np.allclose(loaded_affine, source_affine @ reversal, rtol=0, atol=1e-4):
        return True
    raise ValueError("Loaded image geometry does not match its source voxel grid.")


def roi_geometry(
    affine: NDArray[Any], bounds: list[list[int]]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Convert inclusive source-voxel bounds to an oriented physical box."""
    low, high = np.asarray(bounds, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(affine).all() or np.any(spacing <= 0) or np.any(high < low):
        raise ValueError("ROI geometry is invalid.")
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        raise ValueError("This ROI needs a volume grid with orthogonal voxel axes.")
    transform = np.eye(4)
    transform[:3, :3] = directions
    transform[:3, 3] = (affine @ np.append((low + high) / 2, 1))[:3]
    # Full voxel edges cover both endpoint slices, even for a one-slice ROI.
    return transform, (high - low + 1) * spacing


def source_slice(
    ras_to_ijk: NDArray[Any], xy_to_ras: NDArray[Any], shape: list[int], window: list[float]
) -> dict[str, Any]:
    xy_to_ijk = ras_to_ijk @ xy_to_ras
    # A source axis must be constant over the displayed plane. Oblique slices need
    # resampling and an inverse transform, which this first bridge does not guess.
    normal = np.cross(xy_to_ijk[:3, 0], xy_to_ijk[:3, 1])
    normal /= np.linalg.norm(normal)
    axis = int(np.argmax(np.abs(normal)))
    if abs(normal[axis]) < 0.99999:
        raise ValueError("Align the view with a source voxel plane before slice annotation.")
    index = int(round(xy_to_ijk[axis, 3]))
    if not 0 <= index < shape[axis]:
        raise ValueError("The selected view is outside the source volume.")
    source_axes = [i for i in range(3) if i != axis]
    horizontal, vertical = xy_to_ijk[:3, 0], -xy_to_ijk[:3, 1]
    column_axis, row_axis = int(np.argmax(np.abs(horizontal))), int(np.argmax(np.abs(vertical)))
    if (
        {column_axis, row_axis} != set(source_axes)
        or abs(horizontal[column_axis]) / np.linalg.norm(horizontal) < 0.99999
        or abs(vertical[row_axis]) / np.linalg.norm(vertical) < 0.99999
    ):
        raise ValueError("Align the in-plane view axes with the source grid before annotation.")
    return {
        "axis": axis,
        "index": index,
        "window": window,
        "orientation": {
            "transpose": row_axis != source_axes[0],
            "flip_rows": bool(vertical[row_axis] < 0),
            "flip_columns": bool(horizontal[column_axis] < 0),
        },
    }


def merge_proposal(
    current: NDArray[np.uint8],
    proposed: NDArray[np.uint8],
    labels: list[int],
    scope: dict[str, Any] | None,
) -> NDArray[np.uint8]:
    """Replace only requested labels/voxels; preserve other unsaved local edits."""
    if current.shape != proposed.shape:
        raise ValueError("Proposal shape differs from the source voxel grid.")
    region: list[slice | int] = [slice(None)] * current.ndim
    if scope:
        region[scope["axis"]] = scope["index"]
    result = current.copy()
    target, source = result[tuple(region)], proposed[tuple(region)]
    target[np.isin(target, labels)] = 0
    selected = np.isin(source, labels)
    if np.any(selected & (target != 0)):
        raise ValueError(
            "Proposal overlaps another locally edited structure. Resolve scopes first."
        )
    target[selected] = source[selected]
    return result


def clear_labels(
    current: NDArray[np.uint8], labels: list[int], scope: dict[str, Any] | None
) -> NDArray[np.uint8]:
    """Clear foreground voxels on the source grid without changing the input mask."""
    if not labels or any(type(label) is not int or not 1 <= label <= 255 for label in labels):
        raise ValueError("Choose foreground labels to clear.")
    region: list[slice | int] = [slice(None)] * current.ndim
    if scope is not None:
        axis, index = scope.get("axis"), scope.get("index")
        if (
            current.ndim != 3
            or type(axis) is not int
            or not 0 <= axis < 3
            or type(index) is not int
            or not 0 <= index < current.shape[axis]
        ):
            raise ValueError("Clear request needs a valid source-volume slice.")
        region[axis] = index
    result = current.copy()
    selected = result[tuple(region)]
    selected[np.isin(selected, labels)] = 0
    return result
