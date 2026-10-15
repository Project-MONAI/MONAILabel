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

import multiprocessing
import os
from pathlib import Path

import pytest

from monailabel.totalsegmentator.sockets import worker_sockets


@pytest.mark.skipif(os.name != "posix", reason="Unix socket path limit")
def test_spawned_manager_uses_private_short_socket_not_long_image_spool(tmp_path, monkeypatch):
    long_spool = tmp_path / ("provider-scratch-" + "x" * 120)
    long_spool.mkdir()
    monkeypatch.setenv("TMPDIR", str(long_spool))
    process = multiprocessing.current_process()
    before = process._config.get("tempdir")
    with worker_sockets():
        directory = Path(process._config["tempdir"])
        assert directory.stat().st_mode & 0o777 == 0o700
        with multiprocessing.get_context("spawn").Manager() as manager:
            result = manager.dict()
            result["ok"] = True
            assert result["ok"] is True
            assert manager.address.startswith(str(directory))
        assert os.environ["TMPDIR"] == str(long_spool)
    assert not directory.exists()
    assert process._config.get("tempdir") == before
