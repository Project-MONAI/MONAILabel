"""Pinned local coordinator options. Annotation/training models use a separate registry."""

from dataclasses import dataclass
from typing import Literal

VLLM_IMAGE = (
    "nvcr.io/nvidia/vllm@sha256:fe21f1b1f3a53886515a191ba6309065a54b3e026fe8a43573e75e4ecdfd530d"
)
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:a04d9a1a7ffe371b05230aecab001d4ba2bfa0e5c137bc56409ecc4cbc3ac864"
)


@dataclass(frozen=True)
class LocalModel:
    repository: str
    revision: str
    name: str
    port: int
    engine: Literal["vllm", "sglang"]
    image: str


LOCAL_MODELS = {
    "4b": LocalModel(
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "monailabel-nemotron-4b-dfaf35de",
        8011,
        "vllm",
        VLLM_IMAGE,
    ),
    "9b": LocalModel(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "6533e8de2c68e4536bf7c411d7a3ce5734111476",
        "monailabel-nemotron-9b",
        8012,
        "vllm",
        VLLM_IMAGE,
    ),
    "lightning": LocalModel(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "cc84af2fe71647d87f4486c064f320e1e7535243",
        "monailabel-nemotron-3.5-lightning",
        8013,
        "sglang",
        SGLANG_IMAGE,
    ),
}
