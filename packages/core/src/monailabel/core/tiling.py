"""Native-resolution tiles with context halos and nonoverlapping output ownership."""

from dataclasses import dataclass

from monailabel.core.errors import DomainError
from monailabel.core.models import ImageRegion, ImageTiling


@dataclass(frozen=True)
class ImageTile:
    read: ImageRegion
    write: ImageRegion


def image_tiles(height: int, width: int, config: ImageTiling) -> list[ImageTile]:
    step = config.tile_size - 2 * config.overlap
    count = ((height + step - 1) // step) * ((width + step - 1) // step)
    if count > 4096:
        raise DomainError(
            "This scope needs more than 4,096 tiles. Use a larger tile or smaller region."
        )
    tiles = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            right, bottom = min(x + step, width), min(y + step, height)
            left, top = max(0, x - config.overlap), max(0, y - config.overlap)
            read_right, read_bottom = (
                min(width, right + config.overlap),
                min(height, bottom + config.overlap),
            )
            tiles.append(
                ImageTile(
                    read=ImageRegion(
                        x=left, y=top, width=read_right - left, height=read_bottom - top
                    ),
                    write=ImageRegion(x=x, y=y, width=right - x, height=bottom - y),
                )
            )
    return tiles
