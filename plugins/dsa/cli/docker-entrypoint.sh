#!/usr/bin/env bash

if type nvidia-smi >/dev/null 2>/dev/null;
then
    source /venv-gpu/bin/activate
    echo "NOTE: GPU available" >&2
fi

# Try to start a local version memcached, but fail gracefully if we can't.
memcached -u root -d -m 1024 || true

# Calling the slicer_cli_web.cli_list_entrypoint always works, but we can skip
# an extra exec if we find the path directly
POSSIBLE_PATH="$1/$1.py"
if [[ -f "$POSSIBLE_PATH" ]]; then
    python "$POSSIBLE_PATH" "${@:2}"
else
    python -m slicer_cli_web.cli_list_entrypoint "$@"
fi
