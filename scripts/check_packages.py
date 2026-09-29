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

"""Validate release artifacts and install them with pip in a fresh, disposable environment."""

import argparse
import configparser
import email
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZipFile

from build_package import combined_license

ROOT = Path(__file__).resolve().parents[1]


def check_artifacts(directory: Path) -> str:
    assert (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes() == (
        ROOT / "packages/monailabel/THIRD_PARTY_NOTICES.md"
    ).read_bytes(), "Refresh the packaged third-party notice inventory"
    projects = {
        data["project"]["name"]: (path.parent, data["project"])
        for path in (ROOT / "packages").glob("*/pyproject.toml")
        for data in [tomllib.loads(path.read_text())]
    }
    public = projects["monailabel"][1]
    expected = public["version"]
    assert all(project["version"] == expected for _, project in projects.values())
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Publish one monailabel wheel and one source archive"
    with ZipFile(wheels[0]) as archive, tarfile.open(sources[0]) as source:
        names = archive.namelist()
        metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        dist_info = metadata_paths[0].rsplit("/", 1)[0]
        metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
        root = source.getnames()[0].split("/")[0]
        package_info = source.extractfile(root + "/PKG-INFO")
        assert package_info is not None
        source_metadata = email.message_from_bytes(package_info.read())
        for record in [metadata, source_metadata]:
            assert record["Name"] == "monailabel"
            assert record["Version"] == expected
            assert record["License-Expression"] == combined_license(
                [project["license"] for _, project in projects.values()]
            )
            assert record["Description-Content-Type"] == "text/markdown"
            requirements = record.get_all("Requires-Dist", [])
            assert requirements
            assert not any(r.lower().startswith("monailabel") or "@" in r for r in requirements)
        assert metadata.get_all("Requires-Dist") == source_metadata.get_all("Requires-Dist")
        assert metadata.get_all("License-File") == source_metadata.get_all("License-File")
        for notice in metadata.get_all("License-File", []):
            wheel_notice = archive.read(f"{dist_info}/licenses/{notice}")
            source_notice = source.extractfile(f"{root}/{notice}")
            assert source_notice is not None and source_notice.read() == wheel_notice, notice
            parts = Path(notice).parts
            original = (
                projects[parts[1]][0] / Path(*parts[2:])
                if parts[0] == "licenses"
                else projects["monailabel"][0] / notice
            )
            assert wheel_notice == original.read_bytes(), notice
        for name, (path, project) in projects.items():
            for pattern in project["license-files"]:
                for notice in path.glob(pattern):
                    relative = notice.relative_to(path).as_posix()
                    packaged = relative if name == "monailabel" else f"licenses/{name}/{relative}"
                    assert packaged in metadata.get_all("License-File", []), packaged
        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(archive.read(f"{dist_info}/entry_points.txt").decode())
        expected_scripts = {
            command: entry
            for _, project in projects.values()
            for command, entry in project.get("scripts", {}).items()
        }
        assert dict(entries["console_scripts"]) == expected_scripts
        assert "monailabel/__init__.py" not in names
        count = 0
        for file in names:
            if file.startswith(dist_info + "/") or file.endswith("/"):
                continue
            assert ".dist-info/" not in file and ".data/" not in file, file
            originals = [
                path / "src" / file
                for path, _ in projects.values()
                if (path / "src" / file).is_file()
            ]
            if file == "sam2/upstream.json":
                originals = [ROOT / "packages/sam-runtime/upstream.json"]
            assert len(originals) == 1, file
            content = archive.read(file)
            assert content == originals[0].read_bytes(), file
            member = source.extractfile(f"{root}/src/{file}")
            assert member is not None and member.read() == content, file
            count += 1
        manifest = json.loads(archive.read("sam2/upstream.json"))
        for file, checksum in manifest["files"].items():
            assert hashlib.sha256(archive.read(file)).hexdigest() == checksum, file
        assert not any(n.endswith((".so", ".cu")) for n in names)
        for file in (
            "monailabel/launcher.py",
            "monailabel/server/static/index.html",
            "monailabel/server/resources/assistant/AGENTS.md",
            "monailabel/core/resources/GenericAnatomyColors.LICENSE.txt",
            "monailabel/sam/resources/LICENSE-SAM",
            "monailabel/providers/resources/LICENSE-VISTA3D",
            "monailabel/totalsegmentator/resources/LICENSE-TotalSegmentator",
            "monailabel/nninteractive/worker.py.lock",
        ):
            assert archive.read(file), file
        static = "monailabel/server/static/"
        for notice in ("vendor/LICENSE-markdown-it.txt", "model-icons/LICENSE-MONAI.txt"):
            assert archive.read(static + notice), notice
        unchanged = {
            "vendor/markdown-it-15.0.2.min.mjs": (
                "85feb50fd6ce1b7c49acb02b0337eb622ecc4fc2ed4ae0dcbb8c084556d463b6"
            ),
            "model-icons/monai.ico": (
                "688eda1b17149da490b063300f7836fd5df42004bf6bd5bf1688a6e7596850bc"
            ),
        }
        for file, checksum in unchanged.items():
            assert hashlib.sha256(archive.read(static + file)).hexdigest() == checksum, file
    print(f"One distribution: {count} source files, entry points and license notices verified")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    directory = args.dist.resolve()
    expected = check_artifacts(directory)
    if args.metadata_only:
        print(f"Validated release metadata for MONAI Label {expected}")
        return
    with tempfile.TemporaryDirectory(prefix="monailabel-pip-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        subprocess.run(
            ["uv", "venv", "--seed", "--python", args.python, str(environment)], check=True
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip = [str(python), "-I", "-m", "pip", "--isolated", "--disable-pip-version-check"]
        subprocess.run(
            [
                *pip,
                "install",
                "--index-url",
                "https://pypi.org/simple",
                "--find-links",
                str(directory),
                f"monailabel=={expected}",
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/check_dependencies.py")], cwd=work, check=True
        )
        for command in ("monailabel", "monailabel-server", "monailabel-client"):
            executable = python.parent / command
            if command != "monailabel-client":
                result = subprocess.check_output(
                    [str(executable), "--version"], cwd=work, text=True
                )
                assert result.strip() == f"MONAI Label {expected}", result
            subprocess.run(
                [str(executable), "--help"], cwd=work, check=True, stdout=subprocess.DEVNULL
            )
        subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/check_installed.py")],
            cwd=work,
            check=True,
        )


if __name__ == "__main__":
    main()
