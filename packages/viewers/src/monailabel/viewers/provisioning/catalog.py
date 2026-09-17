"""Pinned declarative release metadata. Updating a viewer does not change installer code."""

import json
import platform
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from pydantic import TypeAdapter


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    system: str
    machine: str
    url: str
    checksum: str
    executable: str
    algorithm: Literal["sha256", "sha512"] = "sha512"
    archive: Literal["tar", "zip"] = "tar"


@dataclass(frozen=True)
class Installation:
    viewer: str
    executable: str
    source: str
    version: str


@dataclass(frozen=True)
class Catalog:
    schema_version: Literal[1]
    releases: list[ToolSpec]
    discovery: dict[str, dict[str, list[str]]]


def catalog() -> Catalog:
    resource = files("monailabel.viewers").joinpath("resources/installers/desktop.json")
    return TypeAdapter(Catalog).validate_python(json.loads(resource.read_text()))


def release(name: str, system: str | None = None, machine: str | None = None) -> ToolSpec | None:
    system = system or platform.system()
    arch = (machine or platform.machine()).casefold()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(arch, arch)
    return next(
        (s for s in catalog().releases if (s.name, s.system, s.machine) == (name, system, arch)),
        None,
    )
