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

"""Check first-party source headers against the MONAI Label Apache-2.0 notice."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE = """Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""
HASH_SUFFIXES = {".py", ".pyi", ".sh", ".bash", ".yaml", ".yml", ".toml", ".conf"}
BLOCK_SUFFIXES = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".css", ".groovy"}
MARKUP_SUFFIXES = {".html", ".xml", ".svg"}


def header(path: Path) -> str | None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    # These files retain upstream notices and exact, independently verified bytes.
    if relative.startswith("packages/sam-runtime/src/") or relative.startswith(
        "packages/server/src/monailabel/server/static/vendor/"
    ):
        return None
    if relative.startswith("packages/sam/src/monailabel/sam/resources/") and path.suffix in {
        ".yaml",
        ".yml",
    }:
        return None
    if path.suffix in HASH_SUFFIXES or "Dockerfile" in path.name:
        return "\n".join(f"# {line}" for line in NOTICE.splitlines()) + "\n"
    if path.suffix in BLOCK_SUFFIXES:
        return f"/*\n{NOTICE}\n*/\n"
    if path.suffix in MARKUP_SUFFIXES:
        return f"<!--\n{NOTICE}\n-->\n"
    return None


def prefix_length(content: str) -> int:
    """Keep interpreter, Docker parser and XML declarations in their required position."""
    offset = 0
    for line in content.splitlines(keepends=True):
        if line.startswith(("#!", "# syntax=", "# escape=", "<?xml ")):
            offset += len(line)
        else:
            break
    return offset


def check(path: Path, *, fix: bool = False) -> bool:
    expected = header(path)
    if expected is None or not path.is_file():
        return True
    content = path.read_text()
    offset = prefix_length(content)
    if content[offset:].lstrip("\n").startswith(expected):
        return True
    if fix:
        # Never overwrite or add our copyright above somebody else's notice.
        opening = content[offset:].splitlines()[:15]
        if any(
            "copyright" in line.casefold() or "SPDX-License-Identifier" in line for line in opening
        ):
            print(f"Review existing license notice: {path.relative_to(ROOT)}")
            return False
        path.write_text(content[:offset] + expected + "\n" + content[offset:].lstrip("\n"))
        return True
    print(f"Missing MONAI license header: {path.relative_to(ROOT)}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Add missing headers")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.files or [
        ROOT / name
        for name in subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
        if name
    ]
    passed = [check(path.resolve(), fix=args.fix) for path in paths]
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
