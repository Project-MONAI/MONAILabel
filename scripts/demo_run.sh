#!/usr/bin/env bash
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

# Run a command in the private installation, retaining all cache locations.
set -euo pipefail
DEMO_RUN_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEMO_RUN_ROOT="$(cd "$DEMO_RUN_SOURCE/.." && pwd -P)"
[[ "$EUID" != 0 && -x "$DEMO_RUN_ROOT/env/bin/python" ]] || {
    printf '%s\n' 'Use the installed, non-root demo environment.' >&2
    exit 1
}
(($#)) || { printf 'Usage: bash scripts/demo_run.sh COMMAND [ARGS...]\n' >&2; exit 1; }
umask 077
export PATH="$DEMO_RUN_ROOT/tools/uv:$DEMO_RUN_ROOT/env/bin:$DEMO_RUN_ROOT/tools/bin:/usr/local/bin:/usr/bin:/bin"
export UV_PROJECT_ENVIRONMENT="$DEMO_RUN_ROOT/env" UV_CACHE_DIR="$DEMO_RUN_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DEMO_RUN_ROOT/tools/python" UV_PYTHON_BIN_DIR="$DEMO_RUN_ROOT/tools/bin"
export UV_TOOL_DIR="$DEMO_RUN_ROOT/tools/uv-tools" UV_TOOL_BIN_DIR="$DEMO_RUN_ROOT/tools/bin"
export UV_PYTHON=3.12 UV_NO_MODIFY_PATH=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$DEMO_RUN_ROOT/cache" TMPDIR="$DEMO_RUN_ROOT/tmp"
export MONAILABEL_DATA_DIR="${MONAILABEL_DATA_DIR:-$DEMO_RUN_ROOT/workspace}"
export MONAILABEL_CACHE_DIR="$DEMO_RUN_ROOT/cache/monailabel"
export MONAILABEL_DATASETS_DIR="$MONAILABEL_CACHE_DIR/datasets"
export MONAILABEL_MODELS_DIR="$MONAILABEL_CACHE_DIR/models"
export MONAILABEL_TOOLS_DIR="$DEMO_RUN_ROOT/tools"
export HF_HOME="$DEMO_RUN_ROOT/cache/huggingface" TORCH_HOME="$DEMO_RUN_ROOT/cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export MONAILABEL_NATIVE_VIEWERS=0 MONAILABEL_DESKTOP_LIMIT=2
export MONAILABEL_DESKTOP_MEMORY=8g MONAILABEL_DESKTOP_CPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4
# Hosted services are intentionally disabled in this local-only pilot.
unset OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY NV_INFERENCE_API_KEY
cd "$DEMO_RUN_SOURCE"
exec "$@"
