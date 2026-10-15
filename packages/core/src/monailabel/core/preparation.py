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

"""Execution capability for host-prepared resources, independent of transport/storage."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from monailabel.core.errors import DomainError

restricted: ContextVar[bool] = ContextVar("monailabel_prepared_models_only", default=False)


def prepared_models_only() -> bool:
    # The environment flag propagates the restriction into fresh inference workers;
    # request/model configuration never controls it.
    return restricted.get() or os.environ.get("MONAILABEL_PREPARED_MODELS_ONLY") == "1"


@contextmanager
def prepared_models(enabled: bool = True) -> Iterator[None]:
    # A nested service cannot relax an inherited restriction.
    token = restricted.set(prepared_models_only() or enabled)
    try:
        yield
    finally:
        restricted.reset(token)


def require_prepared(ready: bool, resource: str) -> None:
    if not ready:
        raise DomainError(
            f"{resource} is not prepared. Ask the demo host to prepare it before use.",
            code="not_prepared",
            status=503,
        )
