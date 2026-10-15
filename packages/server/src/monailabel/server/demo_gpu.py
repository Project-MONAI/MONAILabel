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

"""Opt-in per-process GPU leases for the demo's killable visitor workers.

One workspace has one server owner. The pool does not coordinate other servers,
administrator in-process inference, or arbitrary CUDA programs on the host.
"""

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Lock

from monailabel.core.errors import DomainError

_DEVICE = re.compile(r"(?:0|[1-9][0-9]*|GPU-[0-9a-fA-F-]{8,})\Z")
_POOLS: dict[tuple[str, ...], Queue[str]] = {}
_POOL_LOCK = Lock()


def devices() -> tuple[str, ...]:
    value = os.environ.get("MONAILABEL_DEMO_GPU_DEVICES")
    if value is None:
        return ()
    selected = tuple(item.strip() for item in value.split(","))
    if (
        not 1 <= len(selected) <= 8
        or len(set(selected)) != len(selected)
        or any(not _DEVICE.fullmatch(item) for item in selected)
    ):
        raise DomainError("Invalid host demo GPU configuration.", status=503)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not set(selected) <= {item.strip() for item in visible.split(",")}:
        raise DomainError(
            "Demo GPUs must be included in the host CUDA visibility list.", status=503
        )
    return selected


def pool(selected: tuple[str, ...]) -> Queue[str]:
    # A cached factory alone may run twice on simultaneous cache misses.
    with _POOL_LOCK:
        if selected not in _POOLS:
            slots: Queue[str] = Queue(maxsize=len(selected))
            for device in selected:
                slots.put_nowait(device)
            _POOLS[selected] = slots
        return _POOLS[selected]


@contextmanager
def gpu_slot(check_cancelled: Callable[[], None], *, enabled: bool = True) -> Iterator[str | None]:
    selected = devices() if enabled else ()
    if not selected:
        yield None
        return
    slots = pool(selected)
    while True:
        check_cancelled()
        try:
            device = slots.get(timeout=0.1)
            break
        except Empty:
            continue
    try:
        check_cancelled()
        yield device
    finally:
        slots.put_nowait(device)
