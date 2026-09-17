"""Keep README prompts synchronized with the executable Spleen workflow."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- spleen-prompts:start -->"
END = "<!-- spleen-prompts:end -->"


def render():
    story = json.loads((ROOT / "examples/prompts/spleen.json").read_text())
    return "```text\n" + "\n\n".join(step["prompt"] for step in story["steps"]) + "\n```"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
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
