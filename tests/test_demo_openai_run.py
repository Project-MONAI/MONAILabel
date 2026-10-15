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

import os
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/demo_openai_run.py"))


def test_key_reader_checks_private_file_without_revealing_it(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    key = root / "key"
    key.write_text("sk-disposable-not-an-actual-api-key")
    key.chmod(0o600)
    assert MODULE["read_key"](key).startswith("sk-")
    key.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        MODULE["read_key"](key)
    key.chmod(0o600)
    linked = root / "link"
    linked.symlink_to(key)
    with pytest.raises(OSError):
        MODULE["read_key"](linked)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="directory"):
        MODULE["read_key"](key)


def test_key_reader_rejects_nonregular_and_invalid_files(tmp_path):
    tmp_path.chmod(0o700)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError):
        MODULE["read_key"](fifo)
    key = tmp_path / "key"
    for value in (
        "",
        "sk-short",
        "sk-" + "x" * 4100,
        "sk-" + "x" * 24 + " embedded secret",
        "sk-" + "x" * 24 + " " * 4100,
    ):
        key.write_text(value)
        key.chmod(0o600)
        with pytest.raises(ValueError):
            MODULE["read_key"](key)
