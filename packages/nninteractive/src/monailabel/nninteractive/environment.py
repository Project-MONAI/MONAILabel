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

"""Host-provided, already installed nnInteractive interpreter; no installer invocation."""

import os
from pathlib import Path

from monailabel.core.preparation import require_prepared


def prepared_python() -> Path | None:
    configured = os.environ.get("MONAILABEL_NNINTERACTIVE_PYTHON", "")
    path = Path(configured)
    return path if path.is_absolute() and path.is_file() and os.access(path, os.X_OK) else None


def worker_command() -> list[str]:
    python = prepared_python()
    require_prepared(python is not None, "nnInteractive runtime")
    # Validate installed versions before importing frameworks or loading weights.
    # -I prevents PYTHONPATH/user-site from replacing the prepared runtime.
    return [
        str(python),
        "-I",
        "-c",
        "from importlib.metadata import version; import runpy,sys; "
        "required={'nninteractive':'2.6.0','nnunetv2':'2.8.1','torch':'2.11.0'}; "
        "mismatch=[name for name,wanted in required.items() "
        "if version(name).split('+')[0]!=wanted]; "
        "sys.exit('nnInteractive runtime versions differ from the pinned release') "
        "if mismatch else None; "
        "script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')",
    ]
