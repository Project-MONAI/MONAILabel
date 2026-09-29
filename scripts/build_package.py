# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build one public distribution from the independently packaged workspace projects."""

import argparse
import configparser
import email
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise ValueError(f"Invalid dependency: {requirement}")
    return re.sub(r"[-_.]+", "-", match[0]).lower()


def combined_license(expressions: list[str]) -> str:
    terms = set()
    for expression in expressions:
        if re.search(r"[()]|\bOR\b", expression):
            terms.add(f"({expression})")
        else:
            terms.update(expression.split(" AND "))
    return " AND ".join(sorted(terms))


def write_file(root: Path, name: str, content: bytes) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid archive path: {name}")
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Two components must never silently overwrite each other's files.
    with target.open("xb") as stream:
        stream.write(content)


def assemble(wheels: Path, target: Path, projects: dict) -> None:
    public = dict(projects["monailabel"])
    dependencies = set()
    scripts = {}
    license_files = []
    found = set()
    for path in sorted(wheels.glob("*.whl")):
        with ZipFile(path) as archive:
            names = archive.namelist()
            metadata_path = next(n for n in names if n.endswith(".dist-info/METADATA"))
            metadata = email.message_from_bytes(archive.read(metadata_path))
            name = metadata["Name"]
            assert name in projects and name not in found, name
            found.add(name)
            assert metadata["Version"] == public["version"], name
            assert metadata["Requires-Python"] == public["requires-python"], name
            assert metadata["License-Expression"] == projects[name]["license"], name
            assert not metadata.get_all("Provides-Extra"), "Handle optional dependencies explicitly"
            dist_info = metadata_path.rsplit("/", 1)[0]
            wheel = email.message_from_bytes(archive.read(f"{dist_info}/WHEEL"))
            assert wheel["Root-Is-Purelib"] == "true", "Native components need a platform wheel"
            assert wheel.get_all("Tag") == ["py3-none-any"], name
            for requirement in metadata.get_all("Requires-Dist", []):
                dependency = requirement_name(requirement)
                if dependency in projects:
                    assert requirement == f"{dependency}=={public['version']}", requirement
                else:
                    assert not dependency.startswith("monailabel-"), requirement
                    dependencies.add(requirement)
            entry_points = f"{dist_info}/entry_points.txt"
            if entry_points in names:
                entries = configparser.ConfigParser(interpolation=None)
                entries.optionxform = str
                entries.read_string(archive.read(entry_points).decode())
                assert entries.sections() == ["console_scripts"], name
                for command, entry in entries["console_scripts"].items():
                    assert command not in scripts, command
                    scripts[command] = entry
            for notice in metadata.get_all("License-File", []):
                destination = notice if name == "monailabel" else f"licenses/{name}/{notice}"
                write_file(target, destination, archive.read(f"{dist_info}/licenses/{notice}"))
                license_files.append(destination)
            for file in names:
                if file.startswith(dist_info + "/") or file.endswith("/"):
                    continue
                assert ".data/" not in file and ".dist-info/" not in file, file
                write_file(target / "src", file, archive.read(file))
    assert found == projects.keys(), "Build every workspace component before assembling the release"
    public["dependencies"] = sorted(dependencies)
    public["license"] = combined_license([p["license"] for p in projects.values()])
    public["license-files"] = sorted(license_files)
    urls = public.pop("urls", {})
    public.pop("scripts", None)
    tables = {
        "project": public,
        "project.scripts": scripts,
        "project.urls": urls,
        "build-system": {"requires": ["hatchling>=1.27,<2"], "build-backend": "hatchling.build"},
        "tool.hatch.build.targets.wheel": {
            "packages": [f"src/{p.name}" for p in sorted((target / "src").iterdir())]
        },
        "tool.hatch.build.targets.sdist": {
            "include": [
                "/src",
                "/licenses",
                "/LICENSE",
                "/THIRD_PARTY_NOTICES.md",
                "/README.md",
                "/pyproject.toml",
            ]
        },
    }
    header = Path(__file__).read_text().split('"""', 1)[0]
    (target / "pyproject.toml").write_text(
        header
        + "\n\n".join(
            f"[{name}]\n"
            + "\n".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in table.items())
            for name, table in tables.items()
        )
        + "\n"
    )
    shutil.copy2(ROOT / "packages/monailabel/README.md", target / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    projects = {
        data["project"]["name"]: data["project"]
        for path in sorted((ROOT / "packages").glob("*/pyproject.toml"))
        for data in [tomllib.loads(path.read_text())]
    }
    # Keep old artifacts out of uploads without deleting anyone's builds.
    version = projects["monailabel"]["version"]
    expected = {f"monailabel-{version}-py3-none-any.whl", f"monailabel-{version}.tar.gz"}
    existing = [*destination.glob("*.whl"), *destination.glob("*.tar.gz")]
    if any(path.name not in expected for path in existing):
        parser.error("Use an output directory without other versions or component distributions")
    with tempfile.TemporaryDirectory(prefix="monailabel-release-") as temporary:
        work = Path(temporary)
        wheels = work / "components"
        subprocess.run(
            ["uv", "build", "--all-packages", "--wheel", "--no-sources", "--out-dir", str(wheels)],
            cwd=ROOT,
            check=True,
        )
        source = work / "release"
        assemble(wheels, source, projects)
        # uv builds the wheel from the sdist, verifying that the published source
        # archive is self-contained and needs no workspace or sibling packages.
        subprocess.run(
            ["uv", "build", "--no-sources", "--out-dir", str(destination)],
            cwd=source,
            check=True,
        )


if __name__ == "__main__":
    main()
