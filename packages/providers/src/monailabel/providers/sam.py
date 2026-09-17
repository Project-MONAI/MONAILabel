"""Pinned local SAM capabilities and provenance; no tensor-framework imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamModel:
    name: str
    repository: str
    revision: str
    filename: str
    checksum: str
    size: int
    volume: bool


MODELS = {
    "sam2": SamModel(
        "SAM 2.1",
        "facebook/sam2.1-hiera-tiny",
        "de431c4043854a71d8101e17995dfe596bf101a5",
        "sam2.1_hiera_tiny.pt",
        "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        156008466,
        False,
    ),
    "medsam2": SamModel(
        "MedSAM2",
        "wanglab/MedSAM2",
        "e4a6f35edd7e091619cbc0750f462f1574e23955",
        "MedSAM2_latest.pt",
        "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060",
        156040129,
        True,
    ),
}
