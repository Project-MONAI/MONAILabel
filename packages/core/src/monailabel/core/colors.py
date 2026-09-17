"""Named anatomical display colors with a fallback for non-anatomical labels.

Reference: Slicer's GenericAnatomyColors table, bundled from Slicer 5.12.4.
These are display conventions, not a DICOM requirement or model class IDs.
"""

from functools import lru_cache
from importlib.resources import files

FALLBACK_COLORS = ("#3f9c80", "#d5a654", "#7d85b7", "#ce8175", "#619fb5")


def normalize(name: str) -> str:
    return " ".join(name.casefold().replace("_", " ").replace("-", " ").split())


@lru_cache(maxsize=1)
def anatomical_colors() -> dict[str, str]:
    source = files("monailabel.core").joinpath("resources/GenericAnatomyColors.txt")
    colors = {}
    for line in source.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        _, name, red, green, blue, _ = line.split()
        colors[normalize(name)] = f"#{int(red):02x}{int(green):02x}{int(blue):02x}"
    return colors


def default_color(name: str, identifier: int) -> str:
    if identifier == 0:
        return "#000000"
    name = normalize(name)
    # Models commonly put laterality after the structure name.
    words = name.split()
    if words and words[-1] in {"left", "right"}:
        name = " ".join([words[-1], *words[:-1]])
    name = {
        "kidney": "right kidney",
        "bladder": "urinary bladder",
        "small intestine": "small bowel",
        "gall bladder": "gallbladder",
    }.get(name, name)
    return anatomical_colors().get(name, FALLBACK_COLORS[(identifier - 1) % len(FALLBACK_COLORS)])
