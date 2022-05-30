#!/usr/bin/env bash

# Calling the slicer_cli_web.cli_list_entrypoint always works, but we can skip
# an extra exec if we find the path directly
POSSIBLE_PATH="$1/$1.py"
if [[ -f "$POSSIBLE_PATH" ]]; then
    python "$POSSIBLE_PATH" "${@:2}"
else
    python -m slicer_cli_web.cli_list_entrypoint "$@"
fi
