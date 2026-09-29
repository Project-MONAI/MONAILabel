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

"""Filesystem defaults for the single-workspace server process."""

import os
from pathlib import Path


def workspace_dir() -> Path:
    return Path(os.environ.get("MONAILABEL_DATA_DIR", "workspace")).expanduser()


def configure_workspace(directory: Path | None) -> Path:
    """Set runtime integration paths before providers or viewers are instantiated.

    Explicit cache overrides remain supported. Called by the process entry point,
    not Services, so constructing an isolated test service never changes process paths.
    """
    directory = (directory or workspace_dir()).expanduser().resolve()
    os.environ["MONAILABEL_DATA_DIR"] = str(directory)
    cache = Path(os.environ.setdefault("MONAILABEL_CACHE_DIR", str(directory / ".cache")))
    for key, path in {
        "MONAILABEL_DATASETS_DIR": cache / "datasets",
        "MONAILABEL_MODELS_DIR": cache / "models",
        "MONAILABEL_TOOLS_DIR": cache / "tools",
        "MONAILABEL_QUPATH_DATA": directory / "viewers" / "qupath",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
    }.items():
        os.environ.setdefault(key, str(path.expanduser().resolve()))
    return directory
