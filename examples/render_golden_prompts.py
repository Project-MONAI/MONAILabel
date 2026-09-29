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

"""Keep the extended workflow guide synchronized with executable Spleen prompts."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs/workflows.md"
START = "<!-- spleen-prompts:start -->"
END = "<!-- spleen-prompts:end -->"


def render():
    story = json.loads((ROOT / "examples/prompts/spleen.json").read_text())
    return "\n\n".join("> " + step["prompt"] for step in story["steps"])


def main():
    from quickstart_prompts import cases as quickstart_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    quickstart_cases()
    path = TARGET
    current = path.read_text()
    before, rest = current.split(START, 1)
    _, after = rest.split(END, 1)
    expected = before + START + "\n\n" + render() + "\n\n" + END + after
    if args.check:
        if current != expected:
            raise SystemExit(
                "Golden prompts changed. Run: uv run python examples/render_golden_prompts.py"
            )
        print("Documented golden prompts match their test definitions.")
    else:
        path.write_text(expected)


if __name__ == "__main__":
    main()
