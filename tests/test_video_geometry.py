import io

import numpy as np
import pytest
from PIL import Image

from monailabel.core.errors import DomainError
from monailabel.providers.video_geometry import box_from_mask, mask_png, polygon_from_mask


@pytest.mark.parametrize("region", [(0, 0, 1, 1), (0, 0, 32, 24), (5, 7, 20, 15)])
def test_mask_polygon_source_pixel_edges(region):
    x0, y0, x1, y1 = region
    mask = np.zeros((24, 32), np.uint8)
    mask[y0:y1, x0:x1] = 1
    polygon, complex_shape = polygon_from_mask(mask, 3)
    assert polygon.frame == 3 and polygon.box == list(region)
    assert complex_shape is False
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(mask_png(mask)))), mask > 0)


def test_polygon_conversion_discloses_holes_and_disconnected_regions():
    mask = np.zeros((24, 32), np.uint8)
    mask[2:20, 2:20] = 1
    mask[5:10, 5:10] = 0
    mask[21:23, 25:29] = 1
    polygon, complex_shape = polygon_from_mask(mask, 0)
    assert polygon.box == [2, 2, 20, 20] and complex_shape
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(mask_png(mask)))), mask > 0)
    with pytest.raises(DomainError, match="visible tool"):
        polygon_from_mask(np.zeros_like(mask), 0)


def test_mask_box_keeps_holes_but_rejects_ambiguous_disconnected_tools():
    mask = np.zeros((24, 32), np.uint8)
    mask[2:20, 2:20] = 1
    mask[5:10, 5:10] = 0
    assert box_from_mask(mask, 3).box == [2, 2, 20, 20]
    mask[21:23, 25:29] = 1
    with pytest.raises(DomainError, match="one connected tool"):
        box_from_mask(mask, 3)
