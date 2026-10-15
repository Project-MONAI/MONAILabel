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

"""Short private Unix socket names; model/image scratch remains in the bounded spool."""

import multiprocessing
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast


@contextmanager
def worker_sockets() -> Iterator[None]:
    if os.name != "posix":
        yield
        return
    # multiprocessing copies this config to spawned children. TMPDIR must NOT
    # change: upstream model files still belong in the reserved provider spool.
    config = cast(Any, multiprocessing.current_process())._config
    previous = config.get("tempdir")
    with tempfile.TemporaryDirectory(prefix="ml-ts-", dir="/tmp") as directory:
        config["tempdir"] = directory
        try:
            yield
        finally:
            if previous is None:
                config.pop("tempdir", None)
            else:
                config["tempdir"] = previous
